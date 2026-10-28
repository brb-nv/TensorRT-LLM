# MiniMax-M3.1-Flash on GB300 (run `aria-m31f-sgl-1026`)

This branch is a single TensorRT-LLM tree for MiniMax-M3.1-Flash NVFP4 serving on GB300 (InferenceX AgentX,
agentic-coding, DSpark K=7 MTP, one GB300 node; the TP8 recipes use two 4-GPU trays). Every optimization in this
tree is behind an env switch or a config field, so this one tree with the right env serves every recipe in
[recipes/](recipes/). The exceptions are listed under **Mismatches**.

Base: `brb-nv/TensorRT-LLM` `user/brb/minimax_m3.1_gb300_stack_tp2-v8` @ `705ea94d`. On top of it, in order:

| Commits | What | How it is switched on |
|---|---|---|
| `12fde935a`..`e8fbc085f` | wheel-1: NVFP4 index-K with the MSA-v1 Q8KV4 indexer (C++ producer + Python) | `sparse_attention_config.indexer_kv_dtype: nvfp4` (needs MSA) |
| `cae9c789c`, `29e70b848`, `ed1967d92` | wheel-1: MXFP8 next-layer QKV input from the boundary AllReduce (C++ + Python) | **always on** (bitwise-identical activations) |
| `eca98b6b6` | wheel-1: KV v2 host-tier staging one step ahead | `TRTLLM_M3_ASYNC_OFFLOAD`, `TRTLLM_KV_ONBOARD_PREFETCH*` (default off; no recipe sets them) |
| `7f147b424` | wheel-1: KV v2 layer-wise host-KV onboarding (C++ + Python), onboard grid floor of 32 CTAs on C2C | `TRTLLM_KV_LAYERWISE_ONBOARD` (**default on**; `0` turns it off), `TRTLLM_KV_ONBOARD_CTAS` |
| `46fb32192`, `452b078db` | wheel-2 / H39: AllReduce+RMSNorm NVFP4 linear-SF epilogue, MoE finalize+AR fusion (C++ + Python) | `TRTLLM_ARIA_AR_NVFP4`, `TRTLLM_ARIA_MOE_AR` |
| `f19807875` | S5 overlay (s4stack): never-mix scheduling, fast sampler, decode fusion, prefill GEMM, prefill2 | `TRTLLM_ARIA_NO_MIX*`, `TRTLLM_ARIA_FAST_SAMPLER`, `TRTLLM_ARIA_DECODE_FUSION`, `TRTLLM_ARIA_PREFILL`, `TRTLLM_ARIA_PREFILL2` |
| `1ddc67c6f` | H36: fixed swap-AB CuTeDSL tactic for 33-256-row MXFP8 decode GEMMs | `TRTLLM_ARIA_LOWM_GEMM` |
| this folder | docs, recipes, MSA patch series | - |

The SHAs above are this branch's. The image named in the `repro-` recipes was built before the history was
rewritten for authorship, so its build marker names the pre-rewrite commit `298875d8f`. Its `tensorrt_llm/` and
`cpp/` trees are the same as this branch's.

Other experiments are not on this branch, and no recipe here uses them.

All env switches, their defaults and which recipe sets each one: [env.md](env.md).

## Recipes and stacks

Each recipe in [recipes/](recipes/) is named `c<concurrency>-tp<N>-<stack>-full`. Stacks:

- **stock**: the base image unchanged.
- **S3**: TP4 (or TP8 on two trays), never-mix overlay (`nomix.patch`) + fine CUDA-graph batch list.
- **S4**: wheel-1 image + MSA quad4, NVFP4 index-K, MSA-v1 decode + `nomix.patch`.
- **S5**: wheel-1 image + `s4stack.patch`.
- **S6**: wheel-2 image + `s6.patch`, which is S5 plus H36 and the H39 env switches.
- **+LW**: the same with `TRTLLM_KV_LAYERWISE_ONBOARD=1`.

The TP8 recipes are one TP8 worker on two 4-GPU trays of one NVL72 rack (MNNVL across the trays), with
`modeling_v2: auto`.

The original images map to these files under `MinimaxM3.1_ARIA_SGL/images/`:

- **base**: `trtllm-m31-705ea94.sqsh`. This is the user's `trtllm-m31.sqsh` (CI image + 1113060421 wheel) with the
  705ea94 Python copied over it. It has no MSA. 1113060421..705ea94 changes Python only.
- **wheel-1+MSA-quad4**: `trtllm-m31-705ea94-whl-p17bc18cd20e7-msa1e7bf76-pa20552d.sqsh`. This is a wheel of 705ea94
  + the 11 wheel-1 patches (`7f147b424`'s tree), plus MSA 1e7bf76f + quad4.
- **wheel-2+MSA-quad4**: `trtllm-m31-705ea94-whl-p8fd2745220fc-msa1e7bf76-pa20552d.sqsh`. This is wheel-1 + the 2 H39
  patches (`452b078db`'s tree), plus MSA 1e7bf76f + quad4.

The overlays (`nomix.patch`, `s4stack.patch`, `s6.patch`) are diffs of `tensorrt_llm/**.py` against the image's own
source, mounted over the image's files.

### Source check (how each recipe maps onto this tree)

For each recipe, the source tree its image and overlay produced was rebuilt with `git apply` on a scratch index
(705ea94 + the image's patch series + the overlay). That tree's `tensorrt_llm/` and `cpp/` were then diffed
against this branch:

| Recipes | Image + overlay tree | vs this branch |
|---|---|---|
| all S6 / S6+LW (TP2) | wheel-2 series + `s6.patch` | **identical** (git tree `ce03f039`) |
| S5 (TP2) | wheel-1 series + `s4stack.patch` | branch adds H39 + H36 only. Both are env-gated, default off, and the S5 recipes leave them unset. H39's C++ is additive: a new fusion-op enum, a new op, and new kernel template parameters whose defaults keep the old path. |
| S4 (TP4) | wheel-1 series + `nomix.patch` | branch adds the s4stack levers, H39 and H36, all env-gated, default off, and unset in the S4 recipes. `nomix` itself is identical. |
| S3 / stock (TP4, TP8) | 705ea94 (+ `nomix.patch`) | branch adds everything above. Most of it is gated, but see **Mismatches**. |

### Mismatches

These apply only to the recipes that originally ran on the **base** image: S3 TP4, S3 TP8, and stock TP4.
wheel-1 has three changes that are not switched by the env these recipes set:

1. **Layer-wise host-KV onboarding** (`7f147b424`) is on by default. The `repro-` recipes set
   `TRTLLM_KV_LAYERWISE_ONBOARD=0`, which is what every wheel-1/2 recipe except the `lw` ones also sets.
2. **The onboard copy grid floor of 32 CTAs on C2C links** (`7f147b424`, C++) is always on. The `repro-` recipes set
   `TRTLLM_KV_ONBOARD_CTAS=12`, which is the commit's documented value for the previous GB300 default.
3. **The MXFP8 next-layer QKV input from the boundary AllReduce** (`cae9c789c` + `ed1967d92`) is always on and has no
   switch. It is bitwise identical in activations, so outputs do not change; it removes a standalone quantization
   per layer. (Per [env.md](env.md), this path is skipped when the AllReduce runs as MNNVL, which the TP8 recipes
   may use across trays.)

The MSA package is also present in this branch's image, but these recipes leave `TRTLLM_M3_MSA_V1_DECODE` unset and
use an FP8 index-K, so they never import it.

## Build the wheel and the image

The scripts are part of the aria-op harness (`aria-op/workloads/inferencex/minimaxm31flash-fp4-gb300-trtllm/image/`,
which is site-specific: Slurm account, pyxis, and the user's CI images):

1. **Wheel**: `build-wheel.sbatch` builds `scripts/build_wheel.py --cuda_architectures 103-real` in TRT-LLM's CI image
   (`trtllm-20175.sqsh`). It then force-installs the wheel without dependencies into `trtllm-m31.sqsh` and saves the
   result as `images/trtllm-m31-<commit7>-whl.sqsh`, with `/etc/aria-trtllm-source` = `<repo>@<commit>`.

   ```
   # from aria-op/; COMMIT = this branch's commit (fetch it into ws/trtllm-build/repo first while it is unpushed)
   sbatch --export=ALL,COMMIT=<40-hex sha> workloads/inferencex/minimaxm31flash-fp4-gb300-trtllm/image/build-wheel.sbatch
   ```
2. **MSA layer**: `build-image-msa.sbatch` installs MiniMax MSA into the wheel image. MSA is
   [github.com/MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) (MIT), an **external library** and not part of
   TensorRT-LLM. It goes in at `minimax-msa@1e7bf76f`, with its CUTLASS submodule at the pin. It is installed as
   `pip install --no-deps -e /opt/minimax-msa` on top of the wheel, after the msa-quad4 patch series is applied.
   That series is one patch (sha256 `a20552d3…`): the quad4 Q8KV4 V-scale layout
   (`MINIMAX_MSA_Q8KV4_V_SCALE_LAYOUT=quad4`) that TRT-LLM's `msa_v1_decode` needs. Every image with MSA used exactly
   this series (tag `pa20552d`). MSA has no AOT kernels: it JIT-compiles at first use into `$TORCH_EXTENSIONS_DIR`.

   ```
   sbatch --export=ALL,BASE=<wheel image>,MSA_PATCHES=<msa-quad4 patch dir>,OUT=<out.sqsh>,RUN=aria-m31f-sgl-1026 \
     workloads/inferencex/minimaxm31flash-fp4-gb300-trtllm/image/build-image-msa.sbatch
   ```
   The script then runs `msa_smoke.py` (the quad4 smoke) on the result.

## Run a point

- **As originally run**: [recipes/](recipes/) `<recipe>.yaml` holds each recipe exactly as it was run, with paths
  marked `SITE-SPECIFIC`. It runs on its original image with its overlay:
  `infra/submit.sh aria-m31f-sgl-1026 <recipe> <name> --overlay <overlay>` (aria-op).
- **On this branch**: [recipes/](recipes/) `repro-<recipe>.yaml` uses the same env switches and args on the single
  image `trtllm-m31-frontier-298875d.sqsh`, with no overlay. That image is a clean wheel of this branch's code
  (pre-rewrite commit `298875d8f`) with MSA 1e7bf76f + msa-quad4 on top. The base-image recipes (TP4 and TP8) also
  set the two KV-onboarding env lines described under Mismatches. To run one:
  `infra/submit.sh aria-m31f-sgl-1026 aria/m31f/recipes/repro-<recipe>.yaml <name>`.
- **Multi-node (TP8)**: `c1-tp8-full`, `c2-tp8-full` and their repros put one worker on two nodes (`nodes: 2`,
  `gpus: 8`). aria-op accepts them only when the workload sets `INFERENCEX_MULTINODE=1` (InferenceX's multi-node path,
  `multinode_prepare.py`, multi-node labels); without it the recipe is rejected.
- Outside aria-op, a recipe is a srt-slurm `srtctl` recipe (schema 2). The `benchmark:` block runs InferenceX's
  `srt_agentic.sh` (AgentX agentic-coding trace, aiperf).
