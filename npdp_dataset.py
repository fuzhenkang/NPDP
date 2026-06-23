#!/usr/bin/env python
# coding: utf-8

"""CSV-backed patent dataset used by all NPDP backbones."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


DEFAULT_PROMPT = (
    "Given a patent, Title: {title}\n"
    "Abstract: {abstract}\n"
    "Predict its normalized patent value (between 0 and 1):"
)


class NPDPPatentDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 512,
        title_column: str = "title",
        abstract_column: str = "abstract",
        label_column: str = "target",
        prompt_template: str = DEFAULT_PROMPT,
    ):
        required = {title_column, abstract_column, label_column}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.title_column = title_column
        self.abstract_column = abstract_column
        self.label_column = label_column
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = self.prompt_template.format(
            title=str(row[self.title_column]).replace("\n", " ").strip(),
            abstract=str(row[self.abstract_column]).replace("\n", " ").strip(),
        )
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        encoded["labels"] = torch.tensor(float(row[self.label_column]), dtype=torch.float32)
        return encoded
