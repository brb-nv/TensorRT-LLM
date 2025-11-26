# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for RoPE (Rotary Position Embedding) operations in MLA tests."""

import torch


def rotate_half(x):
    """Rotates half the hidden dims of the input.
    
    Used in RoPE forward transformation.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_inv(x):
    """Inverse rotation of half the hidden dims.
    
    Used in RoPE inverse transformation to recover original values.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


def unembed_rope_values(rope_values, positions, cos_sin_cache):
    """Unembed RoPE values to recover original latent values.
    
    The RoPE embedding applies rotation using cos/sin. This function inverts
    that transformation to get back the original values from the KV cache.
    
    Args:
        rope_values: Embedded RoPE values, shape (world_size - 1, batch_size, rope_dim)
        positions: Position indices for each rank, shape (world_size - 1,)
        cos_sin_cache: Cached cos/sin values, shape (max_pos, rope_dim, 2)
        
    Returns:
        Original unembedded values with same shape as rope_values
    """
    # Ensure float32 for precision
    rope_values = rope_values.to(dtype=torch.float32)
    
    # Get cos/sin for the specified positions
    cos_sin_cache_pos = torch.index_select(cos_sin_cache, 0, positions)
    cos = cos_sin_cache_pos[..., 0].unsqueeze(1)  # (world_size - 1, 1, rope_dim)
    sin = cos_sin_cache_pos[..., 1].unsqueeze(1)  # (world_size - 1, 1, rope_dim)
    
    # Reshape for pairwise rotation
    rope_values_reshaped = (
        rope_values.unflatten(-1, [-1, 2]).transpose(-1, -2).flatten(start_dim=-2)
    )
    
    # Apply inverse RoPE transformation
    orig_rope_values = rope_values_reshaped * cos + rotate_half_inv(rope_values_reshaped) * sin
    
    # Reshape back to original format
    orig_rope_values_reshaped = (
        orig_rope_values.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
    )
    
    return orig_rope_values_reshaped


