#!/usr/bin/env python
# coding: utf-8

"""Shared output-head, pooling, activation, and loss logic for NPDP models."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss, SmoothL1Loss
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


LOSS_CHOICES = ("mse", "l1", "smooth_l1", "bce")
ACTIVATION_CHOICES = ("sigmoid", "identity", "tanh", "relu", "softplus")


def normalize_loss_name(name: Optional[str]) -> str:
    aliases = {
        "mse": "mse",
        "l1": "l1",
        "mae": "l1",
        "smoothl1": "smooth_l1",
        "smooth_l1": "smooth_l1",
        "bce": "bce",
        "bcewithlogitsloss": "bce",
    }
    key = (name or "mse").lower()
    if key not in aliases:
        raise ValueError(f"Unsupported loss: {name}. Choose from {LOSS_CHOICES}.")
    return aliases[key]


def build_activation(name: Optional[str]) -> nn.Module:
    key = (name or "sigmoid").lower()
    activations = {
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
    }
    if key not in activations:
        raise ValueError(f"Unsupported activation: {name}. Choose from {ACTIVATION_CHOICES}.")
    return activations[key]()


def last_valid_token_indices(
    input_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    pad_token_id: Optional[int],
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the absolute index of the final non-padding token for each row."""
    if attention_mask is not None:
        positions = torch.arange(sequence_length, device=device).unsqueeze(0)
        positions = positions.expand(attention_mask.shape[0], -1)
        return positions.masked_fill(attention_mask.to(device) == 0, -1).max(dim=-1).values

    if input_ids is not None and pad_token_id is not None:
        valid = input_ids.to(device).ne(pad_token_id)
        positions = torch.arange(sequence_length, device=device).unsqueeze(0).expand_as(valid)
        return positions.masked_fill(~valid, -1).max(dim=-1).values

    batch_size = input_ids.shape[0] if input_ids is not None else 1
    return torch.full((batch_size,), sequence_length - 1, dtype=torch.long, device=device)


def compute_npdp_loss(
    loss_name: str,
    predictions: torch.Tensor,
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    key = normalize_loss_name(loss_name)
    labels = labels.to(predictions.device).float()
    if key == "bce":
        return BCEWithLogitsLoss()(raw_logits.squeeze(-1), labels.squeeze(-1))
    if key == "l1":
        return L1Loss()(predictions.squeeze(-1), labels.squeeze(-1))
    if key == "smooth_l1":
        return SmoothL1Loss()(predictions.squeeze(-1), labels.squeeze(-1))
    return MSELoss()(predictions.squeeze(-1), labels.squeeze(-1))


class NPDPSequenceRegressionMixin:
    """Reusable regression head for decoder-only language-model backbones."""

    def _init_npdp_head(self, config) -> None:
        self.num_labels = 1
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        self.loss_name = normalize_loss_name(getattr(config, "npdp_loss", "mse"))
        self.activation_name = getattr(config, "npdp_activation", "sigmoid")
        self.activation = build_activation(self.activation_name)

    def _npdp_output(
        self,
        transformer_outputs,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        return_dict: bool,
    ):
        hidden_states = transformer_outputs[0]
        raw_token_logits = self.score(hidden_states)
        token_predictions = self.activation(raw_token_logits)
        indices = last_valid_token_indices(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=getattr(self.config, "pad_token_id", None),
            sequence_length=hidden_states.shape[1],
            device=hidden_states.device,
        )
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        raw_logits = raw_token_logits[batch_indices, indices]
        predictions = token_predictions[batch_indices, indices]
        loss = None
        if labels is not None:
            loss = compute_npdp_loss(self.loss_name, predictions, raw_logits, labels)

        if not return_dict:
            output = (predictions,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=predictions,
            past_key_values=getattr(transformer_outputs, "past_key_values", None),
            hidden_states=getattr(transformer_outputs, "hidden_states", None),
            attentions=getattr(transformer_outputs, "attentions", None),
        )


def configure_npdp(config, loss_name: str = "mse", activation: str = "sigmoid"):
    config.num_labels = 1
    config.problem_type = "regression"
    config.npdp_loss = normalize_loss_name(loss_name)
    config.npdp_activation = activation
    return config
