from typing import Dict, Optional

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.jamba.configuration_jamba import JambaConfig

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.fused_moe import FusedMoE, RenormalizeMoeRoutingMethod
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class JambaAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[JambaConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        # Note: Jamba does not use positional embeddings.
        super().__init__(hidden_size=config.hidden_size,
                         num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         bias=False,
                         rotary_emb=None,
                         pos_embd_params=None,
                         layer_idx=layer_idx,
                         dtype=config.torch_dtype,
                         config=model_config)


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class JambaMambaMixer(nn.Module):

    def __init__(self, config: JambaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size,
                                 self.intermediate_size * 2,
                                 bias=self.use_bias)
        # selective projection used to make dt, B and C input dependent
        self.x_proj = nn.Linear(self.intermediate_size,
                                self.time_step_rank + self.ssm_state_size * 2,
                                bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank,
                                 self.intermediate_size,
                                 bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size,
                                  self.hidden_size,
                                  bias=self.use_bias)

        self.dt_layernorm = RMSNorm(hidden_size=self.time_step_rank,
                                    eps=config.rms_norm_eps,
                                    dtype=config.torch_dtype)
        self.b_layernorm = RMSNorm(hidden_size=self.ssm_state_size,
                                    eps=config.rms_norm_eps,
                                    dtype=config.torch_dtype)
        self.c_layernorm = RMSNorm(hidden_size=self.ssm_state_size,
                                        eps=config.rms_norm_eps,
                                    dtype=config.torch_dtype)

    # fmt: off
    def slow_forward(self, input_states, attention_mask: Optional[torch.LongTensor] = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=dtype
        )
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediate_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediate_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediate_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, intermediate_size, seq_len]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        return self.slow_forward(hidden_states, attention_mask)


class JambaMLP(GatedMLP):

    def __init__(self, model_config: ModelConfig[JambaConfig]):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias if hasattr(config, 'mlp_bias') else False,
            dtype=config.torch_dtype,
            config=model_config,
        )


class JambaMoE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[JambaConfig],
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        # moe gate (linear layer) only runs in half/full precision for now
        self.gate = Linear(self.hidden_dim,
                           self.num_experts,
                           bias=False,
                           dtype=config.torch_dtype,
                           quant_config=None)

        reduce_results = True

        self.experts = FusedMoE(
            num_experts=self.num_experts,
            routing_method=RenormalizeMoeRoutingMethod(top_k=self.top_k),
            hidden_size=self.hidden_dim,
            intermediate_size=self.ffn_dim,
            aux_stream=aux_stream,
            dtype=config.torch_dtype,
            reduce_results=reduce_results,
            model_config=model_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if self.enable_attention_dp and len(all_rank_num_tokens) > 1:
            max_num_token = max(all_rank_num_tokens)
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0, max_num_token - hidden_states.shape[0]))
        router_logits = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits,
                                           all_rank_num_tokens)
        return final_hidden_states


class JambaAttentionDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[JambaConfig], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.self_attn = JambaAttention(model_config, layer_idx)
        self.feed_forward = JambaMLP(model_config)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.pre_ff_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                        eps=config.rms_norm_eps,
                                        dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.pre_ff_layernorm(hidden_states, residual)

        hidden_states = self.feed_forward(hidden_states)

        return hidden_states, residual


class JambaMambaDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[JambaConfig], layer_idx: int,
                 aux_stream: torch.cuda.Stream):
        super().__init__()
        config = model_config.pretrained_config
        self.mamba = JambaMambaMixer(config=config, layer_idx=layer_idx)

        is_moe = (layer_idx -
                  config.expert_layer_offset) % config.expert_layer_period == 0
        if is_moe:
            self.feed_forward = JambaMoE(model_config, aux_stream)
        else:
            self.feed_forward = JambaMLP(model_config)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.pre_ff_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                        eps=config.rms_norm_eps,
                                        dtype=config.torch_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.mamba(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states, residual = self.pre_ff_layernorm(hidden_states, residual)

        hidden_states = self.feed_forward(hidden_states)

        return hidden_states, residual


class JambaModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[JambaConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         config.pad_token_id,
                                         dtype=config.torch_dtype)

        decoder_layers = []
        for layer_idx in range(config.num_hidden_layers):
            is_attention = (layer_idx - config.attn_layer_offset
                            ) % config.attn_layer_period == 0
            if is_attention:
                curr_layer = JambaAttentionDecoderLayer(model_config, layer_idx)
            else:
                curr_layer = JambaMambaDecoderLayer(model_config, layer_idx,
                                                    self.aux_stream)
            decoder_layers.append(curr_layer)
        self.layers = nn.ModuleList(decoder_layers)

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

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

        hidden_states = inputs_embeds

        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("JambaForCausalLM")
class JambaForCausalLM(DecoderModelForCausalLM[JambaModel, JambaConfig]):

    def __init__(self, model_config: ModelConfig[JambaConfig]):
        super().__init__(JambaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict):

        def filter_weights(prefix, weights: Dict):
            result = {}
            for k, v in weights.items():
                if k.startswith(prefix):
                    new_k = k[len(prefix) + 1:]
                    result[new_k] = v
            return result

        params_map = {
            'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        }

        for name, module in self.named_modules():
            if len(module._parameters) > 0:
                names = name.split('.')
                if names[-1] in params_map:
                    module_weights = []
                    for new_name in params_map[names[-1]]:
                        module_weights.append(
                            filter_weights('.'.join(names[:-1] + [new_name]),
                                           weights))
                    module.load_weights(weights=module_weights)
                else:
                    module_weights = filter_weights(name, weights)
                    if hasattr(module, 'load_weights'):
                        module.load_weights(weights=[module_weights])
                    else:
                        for n, p in module.named_parameters():
                            p.data.copy_(module_weights[n][:])
