import math
import os
import numpy as np
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Gemma3TextConfig
from transformers.activations import ACT2FN

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..pipeline_interface import PipelineInterface
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)

trtllm_debug_path = "/home/bbuddharaju/scratch/TensorRT-LLM/trtllm_debug/"

def save_tensor_to_npy(tensorname, tensor):
    numpy_array = tensor.detach().cpu().float().numpy()
    filename = tensorname + ".npy"
    filepath = os.path.join(trtllm_debug_path, filename)
    np.save(filepath, numpy_array)
    # print(f"TRTLLM: {tensorname}:\n{tensor}")

class Gemma3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: Optional[int] = None,
        is_sliding: bool = False,
    ):
        config = model_config.pretrained_config
        rope_params = RopeParams.from_config(config)
        if is_sliding:
            rope_params.theta = 10000
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )
        q_scaling = math.sqrt(config.query_pre_attn_scalar) / math.sqrt(config.head_dim)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            qk_layernorm=True,
            q_scaling=q_scaling,
        )


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dtype = config.torch_dtype
        self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=self.dtype)
        self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=self.dtype)
        self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=self.dtype)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        is_sliding = bool((layer_idx + 1) % config.sliding_window_pattern)
        self.self_attn = Gemma3Attention(
            model_config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
        )

        self.mlp = Gemma3MLP(config)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                 eps=config.rms_norm_eps,
                                                 dtype=config.torch_dtype)
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        print_condition = (self.layer_idx == 0)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if print_condition: save_tensor_to_npy(f"attention_input_{self.layer_idx}", hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        if print_condition: save_tensor_to_npy(f"attention_output_{self.layer_idx}", hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if print_condition: save_tensor_to_npy(f"attention_output_post_layernorm_{self.layer_idx}", hidden_states)
        hidden_states = residual + hidden_states
        if print_condition: save_tensor_to_npy(f"attention_output_with_residual_{self.layer_idx}", hidden_states)
        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        if print_condition: save_tensor_to_npy(f"mlp_input_{self.layer_idx}", hidden_states)
        hidden_states = self.mlp(hidden_states)
        if print_condition: save_tensor_to_npy(f"mlp_output_{self.layer_idx}", hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        if print_condition: save_tensor_to_npy(f"mlp_output_post_layernorm_{self.layer_idx}", hidden_states)
        hidden_states = residual + hidden_states
        if print_condition: save_tensor_to_npy(f"mlp_output_with_residual_{self.layer_idx}", hidden_states)
        return hidden_states


class Gemma3TextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Gemma3TextConfig]):
        super().__init__(model_config)
        config = self.model_config
        self.hidden_size = config.pretrained_config.hidden_size
        self.padding_idx = config.pretrained_config.pad_token_id

        self.embed_tokens = Embedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.pretrained_config.hidden_size,
                            eps=config.pretrained_config.rms_norm_eps,
                            dtype=config.pretrained_config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            save_tensor_to_npy(f"input_embeddings", inputs_embeds)
            inputs_embeds = inputs_embeds * math.sqrt(self.hidden_size)
            save_tensor_to_npy(f"input_embeddings_scaled", inputs_embeds)

        hidden_states = inputs_embeds.to(self.dtype)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(position_ids=position_ids,
                                          hidden_states=hidden_states,
                                          attn_metadata=attn_metadata)

        hidden_states = self.norm(hidden_states)
        save_tensor_to_npy(f"final_norm_output", hidden_states)
        return hidden_states


@register_auto_model("Gemma3ForCausalLM")
class Gemma3ForCausalLM(DecoderModelForCausalLM[Gemma3TextModel,
                                                Gemma3TextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
    ):
        super().__init__(Gemma3TextModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pipeline_interface: Optional[PipelineInterface] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:

        if self._supports_pp and self.pp_size > 1:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pipeline_interface=pipeline_interface,
            )

            # No need to compute logits for non-last PP ranks
            if self.pp_rank < self.pp_size - 1:
                return output
        else:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )
