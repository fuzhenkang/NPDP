#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import copy
import os
from datetime import datetime

import optuna

from npdp_dataset import DEFAULT_PROMPT
from npdp_finetune_common import save_json, train_npdp
from npdp_model_utils import ACTIVATION_CHOICES, LOSS_CHOICES
from npdp_registry import MODEL_REGISTRY, get_model_entry


def parse_csv_ints(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def infer_direction(metric: str):
    lower = metric.lower()
    if any(token in lower for token in ["loss", "mse", "mae", "rmse"]):
        return "minimize"
    return "maximize"


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Optuna hyperparameter search for NPDP models.")
    parser.add_argument("--model_type", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--title_column", default="title")
    parser.add_argument("--abstract_column", default="abstract")
    parser.add_argument("--label_column", default="target")
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT)
    parser.add_argument("--target_modules", default=None)

    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--storage", default=None, help="Optional Optuna storage, for example sqlite:///npdp_optuna.db")
    parser.add_argument("--metric", default="eval_mse", help="Metric key to optimize, for example eval_mse or eval_ndcg@50")
    parser.add_argument("--direction", choices=["minimize", "maximize", "auto"], default="auto")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size_choices", default="1")
    parser.add_argument("--gradient_accumulation_steps_choices", default="8,16,32")
    parser.add_argument("--learning_rate_min", type=float, default=1e-5)
    parser.add_argument("--learning_rate_max", type=float, default=3e-4)
    parser.add_argument("--weight_decay_choices", default="0.0,0.01,0.05")
    parser.add_argument("--warmup_ratio_choices", default="0.03,0.06,0.1")
    parser.add_argument("--max_length_choices", default="256,512,768")
    parser.add_argument("--loss_choices", default="mse")
    parser.add_argument("--activation_choices", default="sigmoid")
    parser.add_argument("--lora_r_choices", default="4,8,16,32")
    parser.add_argument("--lora_alpha_choices", default="16,32,64")
    parser.add_argument("--lora_dropout_min", type=float, default=0.0)
    parser.add_argument("--lora_dropout_max", type=float, default=0.2)
    parser.add_argument("--ndcg_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    return parser.parse_args()


def build_training_args(args, trial: optuna.Trial):
    entry = get_model_entry(args.model_type)
    base_runs_dir = args.runs_dir or os.path.join(
        "runs",
        args.model_type,
        "optuna-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    trial_runs_dir = os.path.join(base_runs_dir, f"trial_{trial.number:03d}")

    loss_choices = parse_csv_strings(args.loss_choices)
    activation_choices = parse_csv_strings(args.activation_choices)
    unknown_losses = sorted(set(loss_choices) - set(LOSS_CHOICES))
    unknown_activations = sorted(set(activation_choices) - set(ACTIVATION_CHOICES))
    if unknown_losses:
        raise ValueError(f"Unsupported loss choices: {unknown_losses}")
    if unknown_activations:
        raise ValueError(f"Unsupported activation choices: {unknown_activations}")

    return argparse.Namespace(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        runs_dir=trial_runs_dir,
        title_column=args.title_column,
        abstract_column=args.abstract_column,
        label_column=args.label_column,
        prompt_template=args.prompt_template,
        max_length=trial.suggest_categorical("max_length", parse_csv_ints(args.max_length_choices)),
        loss=trial.suggest_categorical("loss", loss_choices),
        activation=trial.suggest_categorical("activation", activation_choices),
        epochs=args.epochs,
        batch_size=trial.suggest_categorical("batch_size", parse_csv_ints(args.batch_size_choices)),
        gradient_accumulation_steps=trial.suggest_categorical(
            "gradient_accumulation_steps",
            parse_csv_ints(args.gradient_accumulation_steps_choices),
        ),
        learning_rate=trial.suggest_float(
            "learning_rate",
            args.learning_rate_min,
            args.learning_rate_max,
            log=True,
        ),
        weight_decay=trial.suggest_categorical("weight_decay", parse_csv_floats(args.weight_decay_choices)),
        warmup_ratio=trial.suggest_categorical("warmup_ratio", parse_csv_floats(args.warmup_ratio_choices)),
        ndcg_k=args.ndcg_k,
        seed=args.seed + trial.number,
        fp16=args.fp16,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        lora_r=trial.suggest_categorical("lora_r", parse_csv_ints(args.lora_r_choices)),
        lora_alpha=trial.suggest_categorical("lora_alpha", parse_csv_ints(args.lora_alpha_choices)),
        lora_dropout=trial.suggest_float("lora_dropout", args.lora_dropout_min, args.lora_dropout_max),
        target_modules=args.target_modules or entry["target_modules"],
    )


def main():
    args = parse_args()
    direction = infer_direction(args.metric) if args.direction == "auto" else args.direction
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=direction,
        load_if_exists=args.storage is not None,
    )

    def objective(trial: optuna.Trial):
        training_args = build_training_args(args, trial)
        result = train_npdp(args.model_type, copy.deepcopy(training_args))
        metrics = result["eval_metrics"]
        trial.set_user_attr("output_dir", result["output_dir"])
        trial.set_user_attr("eval_metrics", metrics)
        if args.metric not in metrics:
            available = ", ".join(sorted(metrics))
            raise KeyError(f"Metric {args.metric!r} was not found. Available metrics: {available}")
        return metrics[args.metric]

    study.optimize(objective, n_trials=args.n_trials)

    output_root = args.runs_dir or os.path.join("runs", args.model_type)
    os.makedirs(output_root, exist_ok=True)
    best_path = os.path.join(output_root, "optuna_best_trial.json")
    save_json(
        {
            "study_name": study.study_name,
            "direction": direction,
            "metric": args.metric,
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_output_dir": study.best_trial.user_attrs.get("output_dir"),
            "best_eval_metrics": study.best_trial.user_attrs.get("eval_metrics"),
        },
        best_path,
    )
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best {args.metric}: {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f"Saved best trial summary to {best_path}")


if __name__ == "__main__":
    main()
