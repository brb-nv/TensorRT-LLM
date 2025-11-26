# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simplified MLA Helix distributed testing.

This is a refactored version of test_mla_helix.py with:
- Cleaner separation of concerns
- Extracted helper utilities
- Simplified test scenarios
- Better documentation
- Object-oriented design for test orchestration
"""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping

from .mla_test_helpers import (
    KVCacheSetup,
    LatentCacheGenerator,
    MLADistributedRunner,
    ReferenceMLARunner,
    TestScenario,
)
from .mla_weight_utils import generate_random_weights
from .test_utils.distributed_helpers import run_on_single_rank

# Configure MPI to use cloudpickle for better serialization
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


# ============================================================================
# Test Scenarios
# ============================================================================

# Key test scenarios covering important cases without redundancy
BASIC_SCENARIOS = [
    TestScenario(batch=1, ctx_len=64),        # Tiny context
    TestScenario(batch=1, ctx_len=4096),      # Small context
    TestScenario(batch=1, ctx_len=32768),     # Medium context
    TestScenario(batch=8, ctx_len=4096),      # Multi-batch small
    TestScenario(batch=8, ctx_len=16384),     # Multi-batch medium
]

# Extended scenarios for more thorough testing (not run by default)
EXTENDED_SCENARIOS = [
    TestScenario(batch=1, ctx_len=512),
    TestScenario(batch=1, ctx_len=1024),
    TestScenario(batch=1, ctx_len=2048),
    TestScenario(batch=1, ctx_len=8192),
    TestScenario(batch=1, ctx_len=16384),
    TestScenario(batch=1, ctx_len=65536),
    TestScenario(batch=1, ctx_len=131072),
    TestScenario(batch=8, ctx_len=1024),
    TestScenario(batch=8, ctx_len=2048),
    TestScenario(batch=8, ctx_len=8192),
    TestScenario(batch=8, ctx_len=32768),
    TestScenario(batch=16, ctx_len=1024),
    TestScenario(batch=16, ctx_len=4096),
    TestScenario(batch=16, ctx_len=16384),
]


# ============================================================================
# Helper Functions
# ============================================================================

def create_rope_params(scenario: TestScenario) -> PositionalEmbeddingParams:
    """Create RoPE parameters from test scenario."""
    if scenario.rope_scaling:
        rope_scaling = {
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        }
    else:
        rope_scaling = None
    
    from dataclasses import dataclass
    
    @dataclass
    class RopeConfig:
        hidden_size: int
        num_attention_heads: int
        rope_scaling: dict
        max_position_embeddings: int
        rope_theta: float
        qk_rope_head_dim: int
        model_type: str
    
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=rope_scaling,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )


def create_test_inputs(scenario: TestScenario):
    """Create random test inputs for a scenario."""
    torch.manual_seed(42)
    
    input_ctx = torch.empty(
        scenario.batch * scenario.ctx_len,
        scenario.hidden_size,
        dtype=scenario.dtype,
        device="cuda"
    ).uniform_(-1, 1)
    
    input_gen = torch.empty(
        scenario.batch * scenario.predicted_tokens_per_seq,
        scenario.hidden_size,
        dtype=scenario.dtype,
        device="cuda",
    ).uniform_(-1, 1)
    
    position_ids_ctx = torch.arange(
        scenario.ctx_len, dtype=torch.int, device="cuda"
    ).repeat(scenario.batch)
    
    return input_ctx, input_gen, position_ids_ctx


# ============================================================================
# Main Test Functions
# ============================================================================

@torch.inference_mode
def run_full_test(rank: int, world_size: int, scenario: TestScenario, gen_steps: int):
    """
    Run full MLA Helix distributed test on a single rank.
    
    This function:
    1. Creates test inputs and RoPE configuration
    2. Runs reference single-GPU MLA (on rank 0)
    3. Distributes weights and runs distributed MLA on all ranks
    4. Compares outputs and returns mismatch ratio
    
    Args:
        rank: Current rank
        world_size: Total number of GPUs
        scenario: Test scenario configuration
        gen_steps: Number of generation steps to run
        
    Returns:
        Mismatch ratio (fraction of elements that don't match)
    """
    # Create common setup
    pos_embd_params = create_rope_params(scenario)
    input_ctx, input_gen, position_ids_ctx = create_test_inputs(scenario)
    
    # Create reference MLA model (same on all ranks initially)
    from tensorrt_llm._torch.modules.attention import MLA
    
    mla = MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        enable_unit_test=True,
    ).cuda()
    
    # Initialize weights (same on all ranks due to same seed)
    generate_random_weights(mla)
    weights = mla.state_dict()
    
    # Run reference on rank 0
    ref_output = None
    ref_kv_cache_setup = None
    
    if rank == 0:
        print("Running reference single-GPU MLA...")
        ref_runner = ReferenceMLARunner(scenario, gen_steps)
        ref_output = ref_runner.run(mla, input_ctx, input_gen, position_ids_ctx)
        ref_kv_cache_setup = ref_runner.kv_cache_setup
    else:
        # Create empty tensor on other ranks
        ref_output = torch.empty(
            scenario.ref_steps,
            scenario.batch,
            scenario.hidden_size,
            dtype=scenario.dtype,
            device="cuda",
        )
    
    # Broadcast reference output to all ranks
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={"cp_type": CpType.HELIX}
    )
    ref_output_all = cp_allgather(ref_output, mapping=mapping, dim=0)
    ref_output = ref_output_all.view(world_size, *ref_output.shape)[0]
    
    # Run distributed MLA
    print(f"Rank {rank}: Running distributed MLA...")
    dist_runner = MLADistributedRunner(scenario, rank, world_size, gen_steps)
    
    # Create distributed MLA
    dist_mla = dist_runner.create_mla_model(pos_embd_params)
    dist_runner.load_distributed_weights(dist_mla, weights)
    
    # Setup KV cache
    kv_cache_setup = KVCacheSetup(scenario, dist_runner.mapping, gen_steps)
    dist_runner.extra_attrs["attention_metadata"] = __import__('weakref').ref(
        kv_cache_setup.attn_metadata
    )
    
    # Run context phase
    ctx_len_per_gpu = scenario.ctx_len // world_size
    input_ctx_bs = input_ctx.view(scenario.batch, scenario.ctx_len, scenario.hidden_size)
    input_ctx_rank = input_ctx_bs[:, rank * ctx_len_per_gpu:(rank + 1) * ctx_len_per_gpu, :]
    input_ctx_rank = input_ctx_rank.reshape(
        scenario.batch * ctx_len_per_gpu, scenario.hidden_size
    ).contiguous()
    
    position_ids_ctx_bs = position_ids_ctx.view(scenario.batch, scenario.ctx_len)
    position_ids_ctx_rank = position_ids_ctx_bs[:, rank * ctx_len_per_gpu:(rank + 1) * ctx_len_per_gpu]
    position_ids_ctx_rank = position_ids_ctx_rank.reshape(scenario.batch * ctx_len_per_gpu).contiguous()
    
    from tensorrt_llm._torch.utils import model_extra_attrs
    with model_extra_attrs(dist_runner.extra_attrs):
        dist_mla(position_ids_ctx_rank, input_ctx_rank, kv_cache_setup.attn_metadata)
    
    # Generate latent cache for non-last ranks
    latent_cache_gen = LatentCacheGenerator.generate(
        dist_mla, rank, world_size, ctx_len_per_gpu,
        input_ctx_bs, ref_kv_cache_setup.attn_metadata if rank == 0 else None,
        dist_runner.mapping
    )
    
    # Run generation phase
    outputs = dist_runner.run_generation_steps(
        dist_mla, input_gen, kv_cache_setup, latent_cache_gen
    )
    
    # Compare outputs
    ratio_mismatch = dist_runner.compare_with_reference(outputs, ref_output)
    
    # Cleanup
    kv_cache_setup.shutdown()
    if ref_kv_cache_setup is not None:
        ref_kv_cache_setup.shutdown()
    
    return ratio_mismatch


# ============================================================================
# Pytest Tests
# ============================================================================

@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("scenario", BASIC_SCENARIOS, ids=lambda x: f"batch{x.batch}_ctx{x.ctx_len}")
def test_mla_helix_distributed(scenario: TestScenario, max_mismatch_ratio: float = 0.02):
    """
    Test MLA with Helix context parallelism across 2 GPUs.
    
    This test verifies that the distributed MLA implementation produces
    results that match the single-GPU reference within specified tolerances.
    
    Args:
        scenario: Test scenario configuration
        max_mismatch_ratio: Maximum allowed ratio of mismatched elements (default 2%)
    """
    world_size = 2
    gen_steps = scenario.ref_steps
    
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_on_single_rank,
            *zip(*[(run_full_test, world_size, scenario, gen_steps)] * world_size),
        )
        
        for ratio_mismatch in results:
            assert ratio_mismatch <= max_mismatch_ratio, (
                f"Mismatch ratio {ratio_mismatch:.4f} exceeds threshold {max_mismatch_ratio}"
            )


# ============================================================================
# Performance Benchmark (Optional)
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@pytest.mark.parametrize("scenario", BASIC_SCENARIOS[:3], ids=lambda x: f"batch{x.batch}_ctx{x.ctx_len}")
def test_mla_helix_performance(scenario: TestScenario):
    """
    Benchmark MLA Helix performance (not run by default).
    
    To run: pytest -m benchmark test_mla_helix_improved.py
    """
    world_size = 2
    timing_steps = 256
    gen_steps = scenario.ref_steps + timing_steps
    
    print(f"\n{'='*60}")
    print(f"Performance test: {scenario.batch} batch, {scenario.ctx_len} ctx_len")
    print(f"Running {timing_steps} timed steps...")
    print(f"{'='*60}\n")
    
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = executor.map(
            run_on_single_rank,
            *zip(*[(run_full_test, world_size, scenario, gen_steps)] * world_size),
        )
        
        # Collect results
        mismatch_ratios = list(results)
        
        print(f"\n{'='*60}")
        print(f"Results: Mismatch ratios = {mismatch_ratios}")
        print(f"{'='*60}\n")
        
        # Verify correctness even in performance test
        for ratio in mismatch_ratios:
            assert ratio <= 0.02


# ============================================================================
# Main Entry Point (for direct execution)
# ============================================================================

if __name__ == "__main__":
    """
    Direct execution for manual testing/benchmarking.
    
    Usage: mpirun -n 2 python test_mla_helix_improved.py
    """
    import sys
    
    # Select scenarios to run
    scenarios_to_run = BASIC_SCENARIOS[:3]  # Run first 3 scenarios
    timing_steps = 256
    
    for scenario in scenarios_to_run:
        gen_steps = scenario.ref_steps + timing_steps
        print(f"\n{'='*80}")
        print(f"Running scenario: batch={scenario.batch}, ctx_len={scenario.ctx_len}")
        print(f"Timing {timing_steps} generation steps...")
        print(f"{'='*80}\n")
        
        # Run with MPI
        world_size = 2
        with MPIPoolExecutor(max_workers=world_size) as executor:
            results = executor.map(
                run_on_single_rank,
                *zip(*[(run_full_test, world_size, scenario, gen_steps)] * world_size),
            )
            
            mismatch_ratios = list(results)
            
            if any(m > 0 for m in mismatch_ratios):
                print(f"\n❌ Test FAILED: mismatch ratios = {mismatch_ratios}")
                sys.exit(1)
            else:
                print(f"\n✅ Test PASSED: all outputs match reference")
    
    print(f"\n{'='*80}")
    print("All scenarios completed successfully!")
    print(f"{'='*80}\n")

