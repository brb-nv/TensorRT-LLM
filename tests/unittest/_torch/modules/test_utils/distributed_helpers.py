# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for distributed MLA testing."""

import traceback

import torch

import tensorrt_llm


def copy_weights_for_cp_rank(weights, param_name, dim, rank, world_size):
    """Extract weights slice for a specific CP (context parallel) rank.
    
    Args:
        weights: Dictionary containing model weights
        param_name: Name of the parameter to slice
        dim: Dimension along which to slice
        rank: Current rank
        world_size: Total number of ranks
    """
    w_dim_per_rank = weights[param_name].shape[dim] // world_size
    w_dim_start = rank * w_dim_per_rank
    w_dim_end = w_dim_start + w_dim_per_rank
    slices = [slice(None)] * weights[param_name].ndim
    slices[dim] = slice(w_dim_start, w_dim_end)
    weights[param_name] = weights[param_name][slices]


def run_on_single_rank(func, *args, **kwargs):
    """Run a function on a single rank with proper error handling.
    
    Sets the correct CUDA device for the rank and provides detailed
    error reporting if something goes wrong.
    
    Args:
        func: Function to run
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Result from func
    """
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    print(f"rank {rank} starting")
    try:
        ret = func(rank, *args, **kwargs)
        print(f"rank {rank} done")
        return ret
    except Exception:
        traceback.print_exc()
        tb = traceback.format_exc()
        raise Exception(f"\n\nError occurred on rank {rank}. Original traceback:\n{tb}\n")


