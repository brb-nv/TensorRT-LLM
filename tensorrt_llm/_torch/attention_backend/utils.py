from typing import Optional, Type

from ...models.modeling_utils import QuantConfig
from . import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, MLAParams, PositionalEmbeddingParams
from .trtllm import TrtllmAttention
from .vanilla import VanillaAttention
import torch

def get_attention_backend(backend_name: str) -> Type[AttentionBackend]:
    if backend_name == "VANILLA":
        return VanillaAttention
    elif backend_name == "TRTLLM":
        return TrtllmAttention
    elif backend_name == "FLASHINFER" and IS_FLASHINFER_AVAILABLE:
        from .flashinfer import FlashInferAttention

        return FlashInferAttention
    elif backend_name == "FLASHINFER_STAR_ATTENTION" and IS_FLASHINFER_AVAILABLE:
        from .star_flashinfer import StarAttention

        return StarAttention

    return TrtllmAttention


def create_attention(
    backend_name: str,
    layer_idx: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: Optional[int] = None,
    pos_embd_params: Optional[PositionalEmbeddingParams] = None,
    quant_config: Optional[QuantConfig] = None,
    q_scaling: Optional[float] = None,
    is_mla_enable: bool = False,
    q_lora_rank: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    qk_nope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    predicted_tokens_per_seq: Optional[int] = 1,
    skip_create_weights_in_init: bool = False,
    attention_chunk_size: Optional[int] = None,
):
    if attention_chunk_size is not None and backend_name.upper() != "TRTLLM":
        raise ValueError(
            f"Backend {backend_name} does not support chunked attention.")

    attn_cls = get_attention_backend(backend_name)

    if is_mla_enable:
        assert attn_cls.support_mla(
        ), f"MLA is not supported for {backend_name} backend"
        assert (q_lora_rank > 0 and kv_lora_rank > 0 and qk_rope_head_dim > 0
                and qk_nope_head_dim > 0 and v_head_dim > 0)
        mla_params = MLAParams(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
        )
    else:
        mla_params = None

    return attn_cls(
        layer_idx,
        num_heads,
        head_dim,
        num_kv_heads,
        quant_config=quant_config,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        skip_create_weights_in_init=skip_create_weights_in_init,
        attention_chunk_size=attention_chunk_size,
    )

def create_bidirectional_mm_mask(
    kv_len, ctx_len, mm_token_type_ids
):
    print("CALL TO create_bidirectional_mm_mask.")
    print("kv_len: ", kv_len)
    print("ctx_len: ", ctx_len)
    print("mm_token_type_ids: ", mm_token_type_ids)
    mask_i = torch.tril(
        torch.full((ctx_len, kv_len), True, device="cuda"),
        diagonal=(kv_len - ctx_len),
    )

    # Apply bidirectional mask on images if token type ids are provided.
    if mm_token_type_ids is not None:
        token_type_mask = mm_token_type_ids.unsqueeze(0) == mm_token_type_ids.unsqueeze(1)
        token_type_mask[mm_token_type_ids == 0] = False  # if text token do not change anything.
        token_type_mask = token_type_mask.to('cuda', dtype=torch.bool)

        # Calculate token_type_mask (left, right, top, bottom) to match mask_i's shape.
        # We pad with zeros such that the padded stuff corresponds to text.
        pad_rows = mask_i.shape[0] - token_type_mask.shape[0]
        pad_cols = mask_i.shape[1] - token_type_mask.shape[1]
        padding = (0, pad_cols, 0, pad_rows)
        token_type_mask_padded = torch.nn.functional.pad(token_type_mask, padding, mode='constant', value=0)

        mask_i = token_type_mask_padded | mask_i

    # if attention_mask is not None:
    #     causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
    #     mask_length = attention_mask.shape[-1]

    #     # Then apply padding mask (will mask pad tokens)
    #     padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
    #     padding_mask = padding_mask == 0
    #     causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
    #         padding_mask, False
    #     )

    return mask_i
