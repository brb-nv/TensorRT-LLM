#!/usr/bin/env python3
"""
Compare tensor files between mixedTP2CP2_redo and pureTP4_redo directories.

Mapping (mixedTP2CP2 -> pureTP4):
    rank 0 -> rank 0
    rank 1 -> rank 2
    rank 2 -> rank 1
    rank 3 -> rank 3
"""

import torch
import os
from pathlib import Path

# Directories
mixed_dir = Path("/home/bbuddharaju/scratch/TensorRT-LLM/mixedTP2CP2_redo")
pure_dir = Path("/home/bbuddharaju/scratch/TensorRT-LLM/pureTP4_redo")

# Mapping: mixed_rank -> pure_rank
rank_mapping = {
    0: 0,
    1: 2,
    2: 1,
    3: 3,
}

# File patterns for before_o_proj.pt
# mixedTP2CP2: rank0_cp0_tp0, rank1_cp0_tp1, rank2_cp1_tp0, rank3_cp1_tp1
# pureTP4: rank0_cp0_tp0, rank1_cp0_tp1, rank2_cp0_tp2, rank3_cp0_tp3

mixed_files = {
    0: "rank0_cp0_tp0_before_o_proj.pt",
    1: "rank1_cp0_tp1_before_o_proj.pt",
    2: "rank2_cp1_tp0_before_o_proj.pt",
    3: "rank3_cp1_tp1_before_o_proj.pt",
}

pure_files = {
    0: "rank0_cp0_tp0_before_o_proj.pt",
    1: "rank1_cp0_tp1_before_o_proj.pt",
    2: "rank2_cp0_tp2_before_o_proj.pt",
    3: "rank3_cp0_tp3_before_o_proj.pt",
}

print("=" * 70)
print("Comparing tensors: mixedTP2CP2_redo -> pureTP4_redo")
print("=" * 70)

for mixed_rank, pure_rank in rank_mapping.items():
    mixed_file = mixed_dir / mixed_files[mixed_rank]
    pure_file = pure_dir / pure_files[pure_rank]
    
    print(f"\n[mixed rank {mixed_rank}] -> [pure rank {pure_rank}]")
    print(f"  Mixed file: {mixed_files[mixed_rank]}")
    print(f"  Pure file:  {pure_files[pure_rank]}")
    
    if not mixed_file.exists():
        print(f"  ERROR: Mixed file not found: {mixed_file}")
        continue
    if not pure_file.exists():
        print(f"  ERROR: Pure file not found: {pure_file}")
        continue
    
    # Load tensors
    mixed_tensor = torch.load(mixed_file, map_location='cpu', weights_only=True)
    pure_tensor = torch.load(pure_file, map_location='cpu', weights_only=True)
    
    # Handle case where loaded data might be a dict
    if isinstance(mixed_tensor, dict):
        print(f"  Mixed tensor is a dict with keys: {list(mixed_tensor.keys())}")
        mixed_tensor = list(mixed_tensor.values())[0]
    if isinstance(pure_tensor, dict):
        print(f"  Pure tensor is a dict with keys: {list(pure_tensor.keys())}")
        pure_tensor = list(pure_tensor.values())[0]
    
    print(f"  Mixed shape: {mixed_tensor.shape}, dtype: {mixed_tensor.dtype}")
    print(f"  Pure shape:  {pure_tensor.shape}, dtype: {pure_tensor.dtype}")
    
    if mixed_tensor.shape != pure_tensor.shape:
        print(f"  WARNING: Shape mismatch!")
        continue
    
    # Convert to float for comparison
    mixed_float = mixed_tensor.float()
    pure_float = pure_tensor.float()
    
    # Compute differences
    diff = torch.abs(mixed_float - pure_float)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    # Also compute relative differences where pure_tensor is non-zero
    non_zero_mask = pure_float.abs() > 1e-8
    if non_zero_mask.any():
        rel_diff = diff[non_zero_mask] / pure_float.abs()[non_zero_mask]
        mean_rel_diff = rel_diff.mean().item()
        max_rel_diff = rel_diff.max().item()
    else:
        mean_rel_diff = float('nan')
        max_rel_diff = float('nan')
    
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Max absolute diff:  {max_diff:.6e}")
    print(f"  Mean relative diff: {mean_rel_diff:.6e}")
    print(f"  Max relative diff:  {max_rel_diff:.6e}")

print("\n" + "=" * 70)
print("Comparison complete.")
print("=" * 70)

