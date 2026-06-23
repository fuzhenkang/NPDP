from typing import Optional

import torch
from transformers.models.glm.modeling_glm import GlmModel, GlmPreTrainedModel

from npdp_model_utils import NPDPSequenceRegressionMixin


class GlmForNPDP(NPDPSequenceRegressionMixin, GlmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = GlmModel(config)
        self._init_npdp_head(config)
        self.post_init()

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
