#!/usr/bin/env python
# coding: utf-8

"""Single registry for all supported NPDP backbones."""

from __future__ import annotations

from transformers import AutoConfig


MODEL_REGISTRY = {
    "llama": {"name": "LLaMA", "target_modules": "q_proj,v_proj"},
    "qwen": {"name": "Qwen2", "target_modules": "q_proj,v_proj"},
    "glm": {"name": "GLM", "target_modules": "q_proj,v_proj"},
    "mistral": {"name": "Mistral", "target_modules": "q_proj,v_proj"},
    "baichuan": {
        "name": "Baichuan",
        "target_modules": "W_pack",
        "trust_remote_code": True,
    },
}


def get_model_entry(model_type: str):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from {sorted(MODEL_REGISTRY)}")
    entry = dict(MODEL_REGISTRY[model_type])
    if model_type == "llama":
        from llama_for_npdp import LlamaForNPDP

        entry["model_cls"] = LlamaForNPDP
    elif model_type == "qwen":
        from qwen_for_npdp import QwenForNPDP

        entry["model_cls"] = QwenForNPDP
    elif model_type == "glm":
        from glm_for_npdp import GlmForNPDP

        entry["model_cls"] = GlmForNPDP
    elif model_type == "mistral":
        from mistral_for_npdp import MistralForNPDP

        entry["model_cls"] = MistralForNPDP
    else:
        from baichuan_for_npdp import BaichuanForNPDP

        entry["model_cls"] = BaichuanForNPDP
    return entry


def load_config(checkpoint: str, model_type: str, **kwargs):
    entry = get_model_entry(model_type)
    return AutoConfig.from_pretrained(
        checkpoint,
        trust_remote_code=entry.get("trust_remote_code", False),
        **kwargs,
    )


def load_base_model(checkpoint: str, model_type: str, config, **kwargs):
    entry = get_model_entry(model_type)
    model_cls = entry["model_cls"]
    if model_type == "baichuan":
        return model_cls.from_backbone_pretrained(checkpoint, config=config, **kwargs)
    return model_cls.from_pretrained(
        checkpoint,
        config=config,
        ignore_mismatched_sizes=True,
        **kwargs,
    )
