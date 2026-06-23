#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from npdp_dataset import DEFAULT_PROMPT, NPDPPatentDataset
from npdp_finetune_common import compute_metrics
from npdp_model_utils import configure_npdp
from npdp_registry import get_model_entry, load_base_model, load_config


def read_training_args(model_path):
    path = os.path.join(model_path, "args.json")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_args(model_type):
    parser = argparse.ArgumentParser(description=f"Evaluate {model_type} NPDP model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main(model_type: str):
    args = parse_args(model_type)
    saved = read_training_args(args.model_path)
    entry = get_model_entry(model_type)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path if os.path.exists(os.path.join(args.model_path, "tokenizer_config.json")) else args.checkpoint,
        use_fast=True,
        trust_remote_code=entry.get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = load_config(args.checkpoint, model_type)
    configure_npdp(config, saved.get("loss", "mse"), saved.get("activation", "sigmoid"))
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id
    base_model = load_base_model(
        args.checkpoint,
        model_type,
        config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)

    frame = pd.read_csv(args.data_path)
    dataset = NPDPPatentDataset(
        frame,
        tokenizer,
        max_length=saved.get("max_length", 512),
        title_column=saved.get("title_column", "title"),
        abstract_column=saved.get("abstract_column", "abstract"),
        label_column=saved.get("label_column", "target"),
        prompt_template=saved.get("prompt_template", DEFAULT_PROMPT),
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(args.model_path, "eval_tmp"),
            per_device_eval_batch_size=args.batch_size,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to=[],
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics,
    )
    result = trainer.predict(dataset)
    print(result.metrics)
    if args.output_csv:
        output = frame.copy()
        output["npdp_label"] = result.label_ids.squeeze()
        output["npdp_prediction"] = result.predictions.squeeze()
        output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"Saved predictions to {args.output_csv}")
