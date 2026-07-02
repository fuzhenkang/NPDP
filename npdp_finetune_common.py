#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import ndcg_score
from torch.utils.data import random_split
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from npdp_dataset import DEFAULT_PROMPT, NPDPPatentDataset
from npdp_model_utils import ACTIVATION_CHOICES, LOSS_CHOICES, configure_npdp
from npdp_registry import get_model_entry, load_base_model, load_config


def make_compute_metrics(ndcg_k: int = 20):
    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        predictions = torch.as_tensor(predictions).squeeze()
        labels = torch.as_tensor(labels).squeeze()
        mse = nn.MSELoss()(predictions, labels).item()
        mae = nn.L1Loss()(predictions, labels).item()
        ndcg = -1.0
        if predictions.numel() >= ndcg_k:
            ndcg = float(ndcg_score([labels.numpy()], [predictions.numpy()], k=ndcg_k))
        return {"mse": mse, "mae": mae, f"ndcg@{ndcg_k}": ndcg}

    return compute_metrics


def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    predictions = torch.as_tensor(predictions).squeeze()
    labels = torch.as_tensor(labels).squeeze()
    mse = nn.MSELoss()(predictions, labels).item()
    mae = nn.L1Loss()(predictions, labels).item()
    ndcg = -1.0
    if predictions.numel() >= 20:
        ndcg = float(ndcg_score([labels.numpy()], [predictions.numpy()], k=20))
    return {"mse": mse, "mae": mae, "ndcg@20": ndcg}


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def parse_args(model_type: str):
    entry = get_model_entry(model_type)
    parser = argparse.ArgumentParser(description=f"Fine-tune {entry['name']} for NPDP regression.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--runs_dir", default=None)
    parser.add_argument("--title_column", default="title")
    parser.add_argument("--abstract_column", default="abstract")
    parser.add_argument("--label_column", default="target")
    parser.add_argument("--prompt_template", default=DEFAULT_PROMPT)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--loss", default="mse", choices=LOSS_CHOICES)
    parser.add_argument("--activation", default="sigmoid", choices=ACTIVATION_CHOICES)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--ndcg_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default=entry["target_modules"])
    return parser.parse_args()


def train_npdp(model_type: str, args):
    set_seed(args.seed)
    if args.runs_dir is None:
        args.runs_dir = os.path.join("runs", model_type, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.runs_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.runs_dir, "args.json"))

    entry = get_model_entry(model_type)
    trust_remote_code = entry.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = load_config(args.checkpoint, model_type)
    configure_npdp(config, args.loss, args.activation)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        "quantization_config": quantization_config,
        "device_map": "auto" if quantization_config is not None else None,
    }
    base_model = load_base_model(args.checkpoint, model_type, config, **model_kwargs)
    if quantization_config is not None:
        base_model = prepare_model_for_kbit_training(base_model)
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        modules_to_save=["score"],
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    frame = pd.read_csv(args.data_path)
    dataset = NPDPPatentDataset(
        frame,
        tokenizer,
        max_length=args.max_length,
        title_column=args.title_column,
        abstract_column=args.abstract_column,
        label_column=args.label_column,
        prompt_template=args.prompt_template,
    )
    train_size = int(0.9 * len(dataset))
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    training_args = TrainingArguments(
        output_dir=args.runs_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=make_compute_metrics(args.ndcg_k),
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    output_dir = os.path.join(args.runs_dir, "last")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_json(vars(args), os.path.join(output_dir, "args.json"))
    print(f"Saved NPDP model to {output_dir}")
    return {
        "output_dir": output_dir,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }


def main(model_type: str):
    args = parse_args(model_type)
    return train_npdp(model_type, args)
