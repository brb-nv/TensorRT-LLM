# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for generating random weights for MLA testing."""

from functools import partial

import torch


def init_low_precision(t, op):
    """Initialize low precision tensor by using float32 intermediate."""
    if t.dtype.itemsize <= 1:
        t2 = torch.empty_like(t, dtype=torch.float32)
        op(t2)
        t.copy_(t2)
    else:
        op(t)


def init_uniform(tensor, a=-1.0, b=1.0, use_kaiming=False):
    """Initialize tensor with uniform distribution or Kaiming initialization."""
    if tensor is None:
        return
    if use_kaiming:
        tv = tensor.view(-1, tensor.shape[-2], tensor.shape[-1])
        for t in tv:
            init_low_precision(t, torch.nn.init.kaiming_uniform_)
    else:
        init_low_precision(tensor, partial(torch.nn.init.uniform_, a=a, b=b))


def init_block_scale(tensor, orig_tensor):
    """Initialize block scale tensor based on original tensor."""
    if tensor is None or orig_tensor is None:
        return
    b1, b2 = 128, 128
    orig_tensor = orig_tensor.contiguous().to(tensor.dtype)
    exp1 = (orig_tensor.shape[-2] + b1 - 1) // b1
    exp2 = (orig_tensor.shape[-1] + b2 - 1) // b2
    if tensor.shape[-2] != exp1 or tensor.shape[-1] != exp2:
        b1 = (orig_tensor.shape[-2] + tensor.shape[-2] - 1) // tensor.shape[-2]
        b2 = (orig_tensor.shape[-1] + tensor.shape[-1] - 1) // tensor.shape[-1]
    e1 = orig_tensor.shape[-2] // b1
    e2 = orig_tensor.shape[-1] // b2
    x = orig_tensor[..., : e1 * b1, : e2 * b2].view(*orig_tensor.shape[:-2], e1, b1, e2, b2)
    scale = x.abs().amax(dim=(-3, -1)) / 448.0
    if e1 * b1 != orig_tensor.shape[-2]:
        x2 = orig_tensor[..., e1 * b1 :, : e2 * b2].view(*orig_tensor.shape[:-2], 1, -1, e2, b2)
        scale2 = x2.abs().amax(dim=(-3, -1)) / 448.0
        scale = torch.cat([scale, scale2], dim=-2)
    if e2 * b2 != orig_tensor.shape[-1]:
        x3 = orig_tensor[..., : e1 * b1, e2 * b2 :].view(*orig_tensor.shape[:-2], e1, b1, 1, -1)
        scale3 = x3.abs().amax(dim=(-3, -1)) / 448.0
        if scale.shape[-2] == e1 + 1:
            x4 = orig_tensor[..., e1 * b1 :, e2 * b2 :].view(*orig_tensor.shape[:-2], 1, -1, 1, -1)
            scale4 = x4.abs().amax(dim=(-3, -1)) / 448.0
            scale3 = torch.cat([scale3, scale4], dim=-2)
        scale = torch.cat([scale, scale3], dim=-1)
    tensor.copy_(scale)


def init_linear(mod):
    """Initialize linear module weights."""
    if mod is None:
        return
    init_uniform(mod.weight, use_kaiming=True)
    if hasattr(mod, "weight_scale"):
        init_block_scale(mod.weight_scale, mod.weight)
    if hasattr(mod, "bias"):
        init_uniform(mod.bias)


def generate_random_weights(mla):
    """Generate random weights for MLA module for testing.
    
    Args:
        mla: MLA module instance to initialize weights for
    """
    # Initialize linear modules
    for name in ["kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "o_proj"]:
        init_linear(getattr(mla, name))

    # Initialize RMSNorm modules
    for name in ["kv_a_layernorm", "q_a_layernorm"]:
        if name == "q_a_layernorm":
            mod = getattr(mla, name, None)
        else:
            mod = getattr(mla, name)
        if mod is not None and hasattr(mod, "weight"):
            init_uniform(mod.weight, a=0.9, b=1.1)

    # Initialize k_b_proj_trans
    init_uniform(mla.k_b_proj_trans, use_kaiming=True)
    if hasattr(mla, "k_b_proj_trans_scale"):
        init_block_scale(mla.k_b_proj_trans_scale, mla.k_b_proj_trans)
    
    # Initialize v_b_proj
    init_uniform(mla.v_b_proj)
    if hasattr(mla, "v_b_proj_scale"):
        init_block_scale(mla.v_b_proj_scale, mla.v_b_proj)


