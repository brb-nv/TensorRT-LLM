# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main helper module for MLA Helix distributed testing.

This module provides a cleaner, object-oriented approach to MLA testing.
"""

import time
import weakref
from dataclasses import dataclass
from typing import List, Optional

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionMetadata,
    KVCacheParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.distributed.ops import cp_allgather
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.sampling_params import SamplingParams

from .mla_weight_utils import generate_random_weights
from .test_utils.distributed_helpers import copy_weights_for_cp_rank
from .test_utils.error_reporting import report_tensor_diff
from .test_utils.rope_utils import unembed_rope_values


@dataclass(kw_only=True, frozen=True)
class TestScenario:
    """Configuration for a single MLA test scenario."""
    
    # Model architecture
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 32
    q_lora_rank: int = None
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    hidden_size: int = 2560
    
    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling: bool = False
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    
    # KV cache and execution
    kv_cache_tokens_per_block: int = 32
    predicted_tokens_per_seq: int = 1
    bias: bool = False
    
    # Test dimensions
    batch: int = 8
    ctx_len: int = 1024
    ref_steps: int = 1
    
    # Tolerances
    atol: float = 1e-1
    rtol: float = 5e-2
    
    @property
    def max_position_embeddings(self) -> int:
        return self.ctx_len + 1


class KVCacheSetup:
    """Manages KV cache setup for MLA testing."""
    
    def __init__(self, scenario: TestScenario, mapping: Mapping, gen_steps: int):
        self.scenario = scenario
        self.mapping = mapping
        self.gen_steps = gen_steps
        self.kv_cache_manager = self._create_kv_cache_manager()
        self.attn_metadata = self._create_attn_metadata()
        
    def _create_kv_cache_manager(self) -> KVCacheManager:
        """Create and configure KV cache manager."""
        n_gpu = self.mapping.world_size
        assert self.scenario.ctx_len % n_gpu == 0
        ctx_len_per_gpu = self.scenario.ctx_len // n_gpu
        
        max_tokens = (
            (ctx_len_per_gpu + self.gen_steps + self.scenario.kv_cache_tokens_per_block - 1)
            // self.scenario.kv_cache_tokens_per_block
            * self.scenario.kv_cache_tokens_per_block
            * self.scenario.batch
        )
        
        kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=self.scenario.num_layers,
            num_kv_heads=1,
            head_dim=self.scenario.kv_lora_rank + self.scenario.qk_rope_head_dim,
            tokens_per_block=self.scenario.kv_cache_tokens_per_block,
            max_seq_len=ctx_len_per_gpu + self.gen_steps,
            max_batch_size=self.scenario.batch,
            mapping=self.mapping,
            dtype=str_dtype_to_binding(torch_dtype_to_str(self.scenario.kv_cache_dtype)),
        )
        
        # Add sequences
        for req_id in range(self.scenario.batch):
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=[1] * ctx_len_per_gpu,
                sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
                is_streaming=False,
            )
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            kv_cache_manager.impl.add_sequence(req_id, ctx_len_per_gpu, 1, req)
            req.state = LlmRequestState.GENERATION_IN_PROGRESS
            req.prompt_len = ctx_len_per_gpu
            req.py_prompt_len = req.prompt_len
            
        return kv_cache_manager
    
    def _create_attn_metadata(self) -> AttentionMetadata:
        """Create attention metadata for context phase."""
        ctx_len_per_gpu = self.scenario.ctx_len // self.mapping.world_size
        
        attn_metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([ctx_len_per_gpu] * self.scenario.batch, dtype=torch.int),
            request_ids=list(range(self.scenario.batch)),
            max_num_requests=self.scenario.batch,
            num_contexts=self.scenario.batch,
            prompt_lens=[ctx_len_per_gpu] * self.scenario.batch,
            max_num_tokens=ctx_len_per_gpu,
            kv_cache_manager=self.kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0 for _ in range(self.scenario.batch)],
            ),
            mapping=self.mapping,
        )
        attn_metadata.prepare()
        return attn_metadata
    
    def shutdown(self):
        """Shutdown KV cache manager."""
        self.kv_cache_manager.shutdown()


class LatentCacheGenerator:
    """Generates latent cache for Helix inactive ranks."""
    
    @staticmethod
    def generate(
        mla: MLA,
        rank: int,
        world_size: int,
        ctx_len_per_gpu: int,
        input_ctx_bs: torch.Tensor,
        ref_attn_metadata: Optional[AttentionMetadata],
        mapping: Mapping,
    ) -> Optional[torch.Tensor]:
        """Generate latent cache for generation phase on non-last ranks.
        
        For Helix CP, only the last rank is active during generation. Other ranks
        need to provide cached latent values from their context positions.
        """
        ret = input_ctx_bs.new_empty(
            (world_size - 1, input_ctx_bs.shape[0], mla.kv_lora_rank + mla.qk_rope_head_dim)
        )
        
        if rank == 0:
            assert ref_attn_metadata is not None
            kv_cache_block_offsets = ref_attn_metadata.host_kv_cache_block_offsets
            kv_buffer = ref_attn_metadata.kv_cache_manager.get_buffers(0)
            
            # Get cos/sin cache for inverse RoPE
            _, cos_sin_cache = mla.pos_embd_params.rope.create_rope_const_params()
            cos_sin_cache = cos_sin_cache.reshape(-1, mla.qk_rope_head_dim, 2)
            
            # Extract values from KV cache for each rank
            for r in range(world_size - 1):
                for b in range(input_ctx_bs.shape[0]):
                    block, t = divmod(
                        (r + 1) * ctx_len_per_gpu,
                        ref_attn_metadata.kv_cache_manager.tokens_per_block
                    )
                    kv_block = kv_cache_block_offsets[0, b, 0, block].item()
                    ret[r, b] = kv_buffer[kv_block, 0, t, 0, :]
            
            # Unembed RoPE values
            rope_values = ret[:, :, mla.kv_lora_rank:].clone()
            positions = torch.arange(1, world_size, device=rope_values.device) * ctx_len_per_gpu
            orig_rope_values = unembed_rope_values(rope_values, positions, cos_sin_cache)
            ret[:, :, mla.kv_lora_rank:] = orig_rope_values.to(dtype=ret.dtype)
        
        # Broadcast from rank 0 to all other ranks
        ret_all = cp_allgather(ret, mapping=mapping, dim=0)
        ret = ret_all.view(world_size, *ret.shape)[0]
        
        # Last rank doesn't need latent cache
        if rank == world_size - 1:
            return None
        return ret[rank]


class MLADistributedRunner:
    """Runs distributed MLA testing with Helix context parallelism."""
    
    def __init__(
        self,
        scenario: TestScenario,
        rank: int,
        world_size: int,
        gen_steps: int
    ):
        self.scenario = scenario
        self.rank = rank
        self.world_size = world_size
        self.gen_steps = gen_steps
        self.mapping = Mapping(
            world_size=world_size,
            rank=rank,
            cp_size=world_size,
            cp_config={"cp_type": CpType.HELIX}
        )
        self.ctx_len_per_gpu = scenario.ctx_len // world_size
        
    def create_mla_model(self, pos_embd_params: PositionalEmbeddingParams) -> MLA:
        """Create MLA model for distributed testing."""
        extra_attrs = dict()
        config = ModelConfig(mapping=self.mapping)
        config.extra_attrs = extra_attrs
        
        mla = MLA(
            hidden_size=self.scenario.hidden_size,
            num_attention_heads=self.scenario.num_heads,
            num_key_value_heads=self.scenario.num_kv_heads,
            qk_nope_head_dim=self.scenario.qk_nope_head_dim,
            qk_rope_head_dim=self.scenario.qk_rope_head_dim,
            v_head_dim=self.scenario.v_head_dim,
            q_lora_rank=self.scenario.q_lora_rank,
            kv_lora_rank=self.scenario.kv_lora_rank,
            predicted_tokens_per_seq=self.scenario.predicted_tokens_per_seq,
            max_position_embeddings=self.scenario.max_position_embeddings,
            bias=self.scenario.bias,
            pos_embd_params=pos_embd_params,
            layer_idx=0,
            dtype=self.scenario.dtype,
            config=config,
            enable_unit_test=True,
        ).cuda()
        
        self.extra_attrs = extra_attrs
        return mla
    
    def load_distributed_weights(self, mla: MLA, weights: dict):
        """Load weights distributed across CP ranks."""
        copy_weights_for_cp_rank(weights, "o_proj.weight", 1, self.rank, self.world_size)
        copy_weights_for_cp_rank(weights, "v_b_proj", 0, self.rank, self.world_size)
        mla.load_state_dict(weights)
    
    def run_generation_steps(
        self,
        mla: MLA,
        input_gen: torch.Tensor,
        kv_cache_setup: KVCacheSetup,
        latent_cache_gen: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Run generation steps and collect outputs."""
        position_ids_gen = torch.full(
            (self.scenario.batch,), self.scenario.ctx_len, dtype=torch.int, device="cuda"
        )
        
        outputs = []
        for step in range(self.scenario.ref_steps):
            # Update KV cache tokens
            helix_is_inactive_rank = []
            for req_id in range(self.scenario.batch):
                kv_cache_setup.kv_cache_manager.impl.add_token(req_id)
                if self.rank == self.world_size - 1:
                    helix_is_inactive_rank.append(False)
                    cache_add = step
                else:
                    helix_is_inactive_rank.append(True)
                    cache_add = 0
            
            cached_tokens_per_seq = [
                self.ctx_len_per_gpu + cache_add for _ in range(self.scenario.batch)
            ]
            
            # Create generation metadata
            if step == 0:
                attn_metadata = get_attention_backend("TRTLLM").Metadata(
                    seq_lens=torch.tensor([1] * self.scenario.batch, dtype=torch.int),
                    request_ids=list(range(self.scenario.batch)),
                    max_num_requests=self.scenario.batch,
                    num_contexts=0,
                    prompt_lens=[self.ctx_len_per_gpu] * self.scenario.batch,
                    max_num_tokens=self.ctx_len_per_gpu,
                    kv_cache_manager=kv_cache_setup.kv_cache_manager,
                    kv_cache_params=KVCacheParams(
                        use_cache=True,
                        num_cached_tokens_per_seq=cached_tokens_per_seq,
                    ),
                    enable_context_mla_with_cached_kv=True,
                    helix_is_inactive_rank=torch.tensor(
                        helix_is_inactive_rank, dtype=torch.bool, device="cuda"
                    ),
                )
            else:
                attn_metadata.kv_cache_params = KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=cached_tokens_per_seq,
                )
                attn_metadata.helix_is_inactive_rank = torch.tensor(
                    helix_is_inactive_rank, dtype=torch.bool, device="cuda"
                )
            
            attn_metadata.prepare()
            self.extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
            
            # Run MLA forward
            with model_extra_attrs(self.extra_attrs):
                result = mla(
                    position_ids_gen, input_gen, attn_metadata, latent_cache_gen=latent_cache_gen
                )
            
            outputs.append(result)
            position_ids_gen += 1
            
        return outputs
    
    def compare_with_reference(
        self,
        outputs: List[torch.Tensor],
        ref_output: torch.Tensor
    ) -> float:
        """Compare outputs with reference and return mismatch ratio."""
        output = torch.stack(outputs, dim=0)
        mismatch_count = 0
        
        for ref_step in range(self.scenario.ref_steps):
            for b in range(self.scenario.batch):
                mismatch_count += report_tensor_diff(
                    output[ref_step, b],
                    ref_output[ref_step, b],
                    self.scenario.atol,
                    self.scenario.rtol,
                    f"Rank {self.rank} {self.world_size}-GPU step {ref_step}, batch {b}",
                )
        
        ratio_mismatch = mismatch_count / output.numel()
        print(
            f"Rank {self.rank} {self.world_size}-GPU: "
            f"{mismatch_count}/{output.numel()} mismatches: {ratio_mismatch}"
        )
        return ratio_mismatch


class ReferenceMLARunner:
    """Runs single-GPU reference MLA for comparison."""
    
    def __init__(self, scenario: TestScenario, gen_steps: int):
        self.scenario = scenario
        self.gen_steps = gen_steps
        self.mapping = Mapping(world_size=1, tp_size=1, rank=0)
        
    def run(
        self,
        mla: MLA,
        input_ctx: torch.Tensor,
        input_gen: torch.Tensor,
        position_ids_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Run reference MLA and return outputs."""
        # Setup KV cache
        kv_cache_setup = KVCacheSetup(self.scenario, self.mapping, self.gen_steps)
        
        # Context phase
        mla(position_ids_ctx, input_ctx, kv_cache_setup.attn_metadata)
        
        # Generation phase
        position_ids_gen = torch.full(
            (self.scenario.batch,), self.scenario.ctx_len, dtype=torch.int, device="cuda"
        )
        
        ref_outputs = []
        start = time.time()
        
        for step in range(self.scenario.ref_steps):
            # Update KV cache
            for req_id in range(self.scenario.batch):
                kv_cache_setup.kv_cache_manager.impl.add_token(req_id)
            
            # Create generation metadata
            if step == 0:
                attn_metadata = get_attention_backend("TRTLLM").Metadata(
                    seq_lens=torch.tensor([1] * self.scenario.batch, dtype=torch.int),
                    request_ids=list(range(self.scenario.batch)),
                    max_num_requests=self.scenario.batch,
                    num_contexts=0,
                    prompt_lens=[self.scenario.ctx_len] * self.scenario.batch,
                    max_num_tokens=self.scenario.ctx_len,
                    kv_cache_manager=kv_cache_setup.kv_cache_manager,
                    kv_cache_params=KVCacheParams(
                        use_cache=True,
                        num_cached_tokens_per_seq=[
                            self.scenario.ctx_len + step for _ in range(self.scenario.batch)
                        ],
                    ),
                    enable_context_mla_with_cached_kv=True,
                )
            else:
                attn_metadata.kv_cache_params = KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[
                        self.scenario.ctx_len + step for _ in range(self.scenario.batch)
                    ],
                )
            
            attn_metadata.prepare()
            result = mla(position_ids_gen, input_gen, attn_metadata)
            ref_outputs.append(result)
            position_ids_gen += 1
        
        end = time.time()
        throughput = self.scenario.batch / (end - start) if (end - start) > 0 else float('inf')
        print(f"Reference: {end - start}s, throughput: {throughput} MLA/s")
        
        self.kv_cache_setup = kv_cache_setup  # Store for latent cache generation
        return torch.stack(ref_outputs, dim=0)


