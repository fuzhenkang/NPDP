from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel

from npdp_model_utils import NPDPSequenceRegressionMixin


class BaichuanForNPDP(NPDPSequenceRegressionMixin, PreTrainedModel):
    """NPDP wrapper for Baichuan checkpoints that rely on trusted remote code."""

    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config, backbone=None):
        super().__init__(config)
        self.model = backbone if backbone is not None else AutoModel.from_config(
            config,
            trust_remote_code=True,
        )
        self._init_npdp_head(config)
        nn.init.normal_(self.score.weight, mean=0.0, std=getattr(config, "initializer_range", 0.02))

    @classmethod
    def from_backbone_pretrained(cls, checkpoint, config, **kwargs):
        backbone = AutoModel.from_pretrained(
            checkpoint,
            config=config,
            trust_remote_code=True,
            **kwargs,
        )
        return cls(config=config, backbone=backbone)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        return super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        return self._npdp_output(outputs, input_ids, attention_mask, labels, return_dict)
