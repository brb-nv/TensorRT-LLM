from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.jamba.configuration_jamba import JambaConfig

from tensorrt_llm._torch.attention_backend import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)
from ..modules.decoder_layer import DecoderLayer

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Jamba
class JambaRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class JambaAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[JambaConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        if model_config.fuse_pos_embd:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = None

        super().__init__(hidden_size=config.hidden_size,
                         num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         bias=False,
                         rotary_emb=MixtralRotaryEmbedding(config),
                         pos_embd_params=pos_embd_params,
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

        self.dt_layernorm = JambaRMSNorm(self.time_step_rank,
                                         eps=config.rms_norm_eps)
        self.b_layernorm = JambaRMSNorm(self.ssm_state_size,
                                        eps=config.rms_norm_eps)
        self.c_layernorm = JambaRMSNorm(self.ssm_state_size,
                                        eps=config.rms_norm_eps)

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


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Jamba
class JambaMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size,
                                   self.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(self.hidden_size,
                                 self.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,
                                   self.hidden_size,
                                   bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock with Mistral->Jamba
class JambaSparseMoeBlock(nn.Module):

    def __init__(self, config: JambaConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [JambaMLP(config) for _ in range(self.num_experts)])

    def forward(
            self,
            hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(
                current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class JambaAttentionDecoderLayer(DecoderLayer):

    def __init__(self, config: ModelConfig[JambaConfig], layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]
        self.self_attn = JambaAttention(config, layer_idx)
        self.feed_forward = JambaMLP(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)

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

    def __init__(self, config: ModelConfig[JambaConfig], layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]
        self.mamba = JambaMambaMixer(config=config, layer_idx=layer_idx)

        ffn_layer_class = JambaSparseMoeBlock if num_experts > 1 else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        self.input_layernorm = JambaRMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)

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
            # if :
            #     curr_layer = JambaAttentionDecoderLayer(model_config, layer_idx, self.aux_stream)
            # else:
            #
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
class JambaForCausalLM(DecoderModelForCausalLM[JambaModel,
                                                 JambaConfig]):

    def __init__(self, model_config: ModelConfig[JambaConfig]):
        super().__init__(JambaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)


##########################################################
# TODO:
# 1) Replace MLP
# 2) Replace FusedMoE
# 3) Mamba --> Need to figure the right way to support Mamba
##########################################################
