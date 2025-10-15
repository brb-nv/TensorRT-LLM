import argparse
import os

import yaml


def gen_config_file(work_dir: str,
                    ctx_tp_size: int,
                    ctx_pp_size: int,
                    ctx_cp_size: int,
                    ctx_batch_size: int,
                    ctx_max_num_tokens: int,
                    ctx_max_seq_len: int,
                    ctx_free_gpu_memory_fraction: float,
                    ctx_enable_attention_dp: bool,
                    gen_tp_size: int,
                    gen_pp_size: int,
                    gen_cp_size: int,
                    gen_moe_ep_size: int,
                    gen_batch_size: int,
                    gen_max_num_tokens: int,
                    gen_max_seq_len: int,
                    gen_enable_attention_dp: bool,
                    gen_gpu_memory_fraction: float,
                    eplb_num_slots: int,
                    mtp_size: int = 0,
                    cache_transceiver_max_num_tokens: int = 4608,
                    model_path: str = "") -> None:
    """
    Generate configuration YAML file for disaggregated inference.

    Args:
        config_path: Path to save the config file
        model_path: Path to the model
        num_ctx_servers: Number of context servers
        ctx_tp_size: Tensor parallel size for context servers
        ctx_pp_size: Pipeline parallel size for context servers
        ctx_cp_size: Context parallel size for context servers
        ctx_batch_size: Batch size for context servers
        ctx_max_num_tokens: Max number of tokens for context servers
        ctx_max_seq_len: Max sequence length for context servers
        ctx_free_gpu_memory_fraction: Free GPU memory fraction for context servers
        ctx_enable_attention_dp: Enable attention DP for context servers
        num_gen_servers: Number of generation servers
        gen_tp_size: Tensor parallel size for generation servers
        gen_pp_size: Pipeline parallel size for generation servers
        gen_cp_size: Context parallel size for generation servers
        gen_batch_size: Batch size for generation servers
        gen_max_num_tokens: Max number of tokens for generation servers
        gen_enable_attention_dp: Enable attention DP for generation servers
        gen_gpu_memory_fraction: GPU memory fraction for generation servers
        eplb_num_slots: Number of slots for eplb
        worker_start_port: Start port for workers
        server_port: Server port
    """
    # Source: docs/source/deployment-guide/quick-start-recipe-for-deepseek-r1-on-trtllm.md
    # Support matrix for moe_backend.
    # | device | Checkpoint | Supported moe_backend |
    # |----------|----------|----------|
    # | H100/H200 | FP8 | CUTLASS |
    # | B200/GB200 EP<=8 | NVFP4 | CUTLASS, TRTLLM |
    # | B200/GB200 EP<=8 | FP8 | DEEPGEMM |
    # | GB200 NVL72 EP>8 | NVFP4 |  WIDEEP |
    # | GB200 NVL72 EP>8 | FP8 | N/A (WIP) |

    ctx_moe_ep_size = ctx_cp_size * ctx_tp_size
    assert gen_enable_attention_dp is False, "Let's keep it simple for now."
    if 'fp4' in model_path or 'FP4' in model_path:
        if ctx_moe_ep_size <= 8:
            ctx_moe_backend = "TRTLLM"
        else:
            print(
                f"Only WideEP supports EP>8 for FP4 with ctx_moe_ep_size: {ctx_moe_ep_size}. But, that needs AttentionDP. So, keeping moe_ep_size = 8 and ctx_moe_backend = TRTLLM."
            )
            ctx_moe_backend = "TRTLLM"
            ctx_moe_ep_size = 8
    elif 'fp8' in model_path or 'FP8' in model_path:
        if ctx_moe_ep_size <= 8:
            ctx_moe_backend = "DEEPGEMM"
        else:
            raise ValueError(
                f"No existing moe support for FP8 with ctx_moe_ep_size: {ctx_moe_ep_size}."
            )
    else:
        raise ValueError(
            f"Can't determine model dtype from model path: {model_path}")

    ctx_config = {
        'build_config': {
            'max_batch_size': ctx_batch_size,
            'max_num_tokens': ctx_max_num_tokens,
            'max_seq_len': ctx_max_seq_len,
        },
        'max_batch_size': ctx_batch_size,
        'max_num_tokens': ctx_max_num_tokens,
        'max_seq_len': ctx_max_seq_len,
        # @B: Enable CUDA graphs later (note: for context in helix, this is not important).
        'cuda_graph_config': None,
        'tensor_parallel_size': ctx_tp_size,
        'moe_expert_parallel_size': ctx_moe_ep_size,
        'enable_attention_dp': True if ctx_enable_attention_dp else False,
        'pipeline_parallel_size': ctx_pp_size,
        'context_parallel_size': ctx_cp_size,
        'print_iter_log': True,
        'disable_overlap_scheduler': True,
        'kv_cache_config': {
            'enable_block_reuse': False,
            'free_gpu_memory_fraction': ctx_free_gpu_memory_fraction,
            'dtype': 'fp8',
        },
        'moe_config': {
            'backend': ctx_moe_backend,
        },
        'cache_transceiver_config': {
            'max_tokens_in_buffer': cache_transceiver_max_num_tokens,
            'backend': 'UCX',
        },
    }

    # Assert that gen_batch_size is a power of 2.
    assert gen_batch_size > 0 and (gen_batch_size & (gen_batch_size - 1)) == 0, \
        f"gen_batch_size ({gen_batch_size}) must be a power of 2."

    gen_config = {
        'build_config': {
            'max_batch_size': gen_batch_size,
            'max_num_tokens': gen_max_num_tokens,
            'max_seq_len': gen_max_seq_len,
        },
        'tensor_parallel_size': gen_tp_size,
        'moe_expert_parallel_size': gen_moe_ep_size,
        'enable_attention_dp': True if gen_enable_attention_dp else False,
        'pipeline_parallel_size': gen_pp_size,
        'context_parallel_size': gen_cp_size,
        'max_batch_size': gen_batch_size,
        'max_num_tokens': gen_max_num_tokens,
        'max_seq_len': gen_max_seq_len,
        'cuda_graph_config': {},
        'print_iter_log': True,
        'kv_cache_config': {
            'enable_block_reuse': False,
            'free_gpu_memory_fraction': gen_gpu_memory_fraction,
            'dtype': 'fp8',
        },
        'moe_config': {
            'backend': 'TRTLLM',
        },
        'cache_transceiver_config': {
            'max_tokens_in_buffer': cache_transceiver_max_num_tokens,
            'backend': 'UCX',
        },
        'stream_interval': 20,
    }

    if gen_tp_size == 8 and not gen_enable_attention_dp:
        gen_config['allreduce_strategy'] = "MNNVL"

    if eplb_num_slots > 0:
        moe_load_balancer_file = os.path.join(work_dir,
                                              "moe_load_balancer.yaml")
        moe_load_balancer_config = {
            'num_slots': eplb_num_slots,
            'layer_updates_per_iter': 1
        }
        with open(moe_load_balancer_file, "w") as f:
            yaml.dump(moe_load_balancer_config,
                      f,
                      default_flow_style=False,
                      sort_keys=False)
        gen_config['moe_config']['load_balancer'] = moe_load_balancer_file

    if mtp_size > 0:
        ctx_config['speculative_config'] = {
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': mtp_size
        }
        gen_config['speculative_config'] = {
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': mtp_size
        }

    ctx_config_file = os.path.join(work_dir, "ctx_config.yaml")
    gen_config_file = os.path.join(work_dir, "gen_config.yaml")
    with open(ctx_config_file, "w") as f:
        yaml.dump(ctx_config, f, default_flow_style=False, sort_keys=False)
    with open(gen_config_file, "w") as f:
        yaml.dump(gen_config, f, default_flow_style=False, sort_keys=False)

    print(
        f"ctx_config_file: {ctx_config_file} gen_config_file: {gen_config_file} generated successfully"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_dir",
                        type=str,
                        default="logs",
                        help="Work directory")
    parser.add_argument("--ctx_tp_size",
                        type=int,
                        default=4,
                        help="Tensor parallel size for context servers")
    parser.add_argument("--ctx_pp_size",
                        type=int,
                        default=1,
                        help="Pipeline parallel size for context servers")
    parser.add_argument("--ctx_cp_size",
                        type=int,
                        default=1,
                        help="Context parallel size for context servers")
    parser.add_argument("--ctx_batch_size",
                        type=int,
                        default=1,
                        help="Batch size for context servers")
    parser.add_argument("--ctx_max_num_tokens",
                        type=int,
                        default=8192,
                        help="Max number of tokens for context servers")
    parser.add_argument("--ctx_max_seq_len",
                        type=int,
                        default=8192,
                        help="Max sequence length for context servers")
    parser.add_argument("--ctx_free_gpu_memory_fraction",
                        type=float,
                        default=0.75,
                        help="Free GPU memory fraction for context servers")
    parser.add_argument("--ctx_enable_attention_dp",
                        dest='ctx_enable_attention_dp',
                        action='store_true',
                        help="Enable attention DP for context servers")
    parser.add_argument("--gen_tp_size",
                        type=int,
                        default=8,
                        help="Tensor parallel size for generation servers")
    parser.add_argument("--gen_pp_size",
                        type=int,
                        default=1,
                        help="Pipeline parallel size for generation servers")
    parser.add_argument("--gen_cp_size",
                        type=int,
                        default=1,
                        help="Context parallel size for generation servers")
    parser.add_argument("--gen_moe_ep_size",
                        type=int,
                        default=1,
                        help="MOE expert parallel size for generation servers")
    parser.add_argument("--gen_batch_size",
                        type=int,
                        default=256,
                        help="Batch size for generation servers")
    parser.add_argument("--gen_max_num_tokens",
                        type=int,
                        default=256,
                        help="Max number of tokens for generation servers")
    parser.add_argument("--gen_max_seq_len",
                        type=int,
                        default=9216,
                        help="Max sequence length for generation servers")
    parser.add_argument("--gen_enable_attention_dp",
                        dest='gen_enable_attention_dp',
                        action='store_true',
                        help="Enable attention DP for generation servers")
    parser.add_argument("--gen_gpu_memory_fraction",
                        type=float,
                        default=0.8,
                        help="GPU memory fraction for generation servers")
    parser.add_argument("--eplb_num_slots",
                        type=int,
                        default=0,
                        help="Number of slots for eplb")
    parser.add_argument("--mtp_size",
                        type=int,
                        default=0,
                        help="Number of nextn layers for MTP")
    parser.add_argument("--cache_transceiver_max_num_tokens",
                        type=int,
                        default=8448,
                        help="Max number of tokens for cache transceiver")
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    args = parser.parse_args()

    gen_config_file(
        args.work_dir, args.ctx_tp_size, args.ctx_pp_size, args.ctx_cp_size,
        args.ctx_batch_size, args.ctx_max_num_tokens, args.ctx_max_seq_len,
        args.ctx_free_gpu_memory_fraction, args.ctx_enable_attention_dp,
        args.gen_tp_size, args.gen_pp_size, args.gen_cp_size,
        args.gen_moe_ep_size, args.gen_batch_size, args.gen_max_num_tokens,
        args.gen_max_seq_len, args.gen_enable_attention_dp,
        args.gen_gpu_memory_fraction, args.eplb_num_slots, args.mtp_size,
        args.cache_transceiver_max_num_tokens, args.model_path)
