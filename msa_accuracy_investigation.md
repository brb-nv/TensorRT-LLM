# MiniMax-M3 MSA Backend Accuracy Investigation

Debugging the accuracy regression on the MSA-backed sparse-attention path
(`use_msa=True`) of `TestMiniMaxM3MXFP8::test_mxfp8`, while the in-tree Triton
reference path (`use_msa=False`) passes.

- **Test:** `tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestMiniMaxM3MXFP8::test_mxfp8`
- **Config:** `tp_size=ep_size=4`, `max_seq_len=4096`, MXFP8 checkpoint, BF16 KV cache.
- **MSA kernels** (`fmha_sm100`): vendored from `/home/scratch.bbuddharaju_gpu/msa`, brought in with commit `f3a5563459` (TRT-LLM backend integration).
- **Models:** `/home/scratch.trt_llm_data_ci/llm-models/MiniMax-M3-MXFP8`.

---

## Model geometry (from checkpoint `config.json`)

| Field | Value |
|---|---|
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 4 |
| `head_dim` | 128 |
| `sparse_num_index_heads` | 4 |
| `sparse_topk_blocks` | 16 |
| `sparse_block_size` | 128 |
| `sparse_init_block` / `sparse_local_block` | 0 / 1 |
| `sparse_disable_index_value` | 1 on sparse layers (3..59) |

**Global:** `num_index_heads (4) == num_kv_heads (4)` → one index head per GQA
group (matches the MSA paper eq. 7 and SGLang).

**Per rank at tp=4:** KV heads are sharded (`num_kv_heads=1` per rank) but the
index projection (`index_q_proj`) is **replicated** (`num_index_heads=4` per
rank). So each rank sees `group = num_index_heads / num_kv_heads = 4`. This
`group>1` regime is a tensor-parallel artifact and is central to the analysis.

---

## Methodology: staged parity test

New diagnostic suite:
`tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_vs_triton_parity.py`

The MSA sparse pipeline is split into independently-checkable stages so the
*first failing stage localizes the bug*, rather than assuming any one component
(e.g. the indexer) is at fault:

| Stage | Test | Isolates | Needs SM100 + fmha_sm100 |
|---|---|---|---|
| 0 | `selection_semantics_spec` | union vs amax math (pure torch) | no |
| 1a | `build_kv_indices_global_page_table` | `kv_indices` page table (pure CPU) | no |
| 1 | `index_scoring_perhead_topk_parity` | proxy FMHA scoring (cache permute, `kv_indices`, causal offset, scale, matmul) | yes |
| 1c | `sparse_topk_select_kernel_vs_torch` / `..._num_valid_pages_contract` | the `sparse_topk_select` kernel in isolation | yes |
| 2 | `block_selection_parity` | full selection incl. init/local forcing + reduction | yes |
| 2b | `block_selection_parity_no_forcing` | selection with forcing OFF (isolates forcing from reduction) | yes |
| 3 | `main_attention_dense_equivalence` | sparse GQA math + HND cache permute + causal, selection factored out | yes |
| 4 | `full_layer_output_parity` | end-to-end MSA vs Triton | yes |

Stages run for both a `group1` (num_kv_heads=4, num_index_heads=4) and a
`group4` (num_kv_heads=1, num_index_heads=4, mirrors per-rank tp=4) config.
Selection stages use long, mixed-length sequences (`[2600, 4000, 800]` →
`[21, 32, 7]` blocks) so top-k actually drops blocks; short sequences masked
several bugs because with `n_valid_blocks <= topk` the selection is
score-independent.

---

## Findings and root causes

### Bug #1 — corrupted `kv_indices` page table (FIXED, commit `677bcb45e5`)

`_build_kv_indices_and_lens` clamped page ids with
`clamp_max(max_page - 1)` where `max_page = max_kv_len // page_size` is the
**per-request** page count — but the page ids read from `req_to_token` are
**global** ids into the paged KV cache. This collapsed the page table for every
request whose pages exceed that bound: every request after the first in a
contiguous layout, and virtually all requests in production (block ids are
global and non-contiguous). The MSA proxy FMHA then read the wrong K/V →
corrupted per-block scores → wrong selection.

**Evidence:** with long sequences, Stage 1 failed `4/12` decode pairs = exactly
`q=1` (the 4000-token / 32-block request) across all 4 heads; `q=0` (21 blocks,
pages `[0..20]`, unclamped) passed; `q=2` (7 blocks ≤ topk) passed because the
selection is score-independent when `n_valid_blocks <= topk`. The kernel-only
Stage 1c passed, ruling out `sparse_topk_select`.

**Why it was masked earlier:** short-sequence runs had every request ≤ topk
blocks, so corrupted scores never changed the selection.

**Fix:** remove the erroneous clamp; ids from `req_to_token` are valid global
ids by construction. Also restores the pre-existing
`test_build_kv_indices_packs_per_request_pages` expectation
(`[0, 1, 3, 4, 5]`).

### Bug #2 — batch-wide scalar forcing in `sparse_topk_select` (FIXED, commit `92d8405af5`)

`sparse_topk_select` only accepts a **scalar** `num_valid_pages` and scalar
`force_begin_blocks` / `force_end_blocks`, applied uniformly to every
(head, token) row. `_msa_index_proxy_and_topk` passed `num_valid_pages =
batch-max` and `force_end_blocks = local_blocks`, so the forced local block was
pinned to the global `nvp-1` instead of each query's own last valid block, and
the OOB clamp used a batch-wide bound. Wrong for any request shorter than the
batch-longest, and for every prefill query token (each has its own causal
extent).

**Evidence:** Stage 2 (with forcing) failed even at `group=1` (`8/12` decode =
queries 0 and 2; query 1 = the batch-longest, passed). Stage 2b (forcing off)
passed at `group=1`, isolating the failure to the forcing path.

**Fix:** replace the kernel selection with a per-query torch selection
(`_select_blocks_from_maxscore` + `_per_token_valid_blocks`) that mirrors the
reference `_index_attention_and_select` (init/local forcing + per-query
valid-block masking + top-k) on the amax-reduced per-KV-head scores. The MSA
path already disables CUDA graphs, so host-side selection is safe. The
`sparse_topk_select` kernel is left available for a future per-query-capable
selector.

### Bug #3 — amax-vs-union index-head reduction (OPEN, NEEDS REVISIT)

For `group>1`, MSA reduces the per-index-head block scores with `amax` then
takes a single top-16, whereas the reference takes per-index-head top-16 then
**unions** across the group. The amax result is a strict **subset** of the
union.

**Evidence:** after bugs #1+#2, Stage 2b `group4` shows MSA ⊂ Triton with
`only_msa=[]` in every case (e.g. Triton picks ~20–30 blocks, MSA picks ≤16, all
contained in Triton's set); `1200/1600` prefill pairs, `2/3` decode.

**Nuance — neither path is the "true" model under TP:** globally M3 has one
index head per GQA group. Under tp=4 the index heads are replicated but KV
heads sharded, so the correct per-rank selection uses *only the index head
matching the rank's KV head* (`group=1`, top-16). The Triton union is an
accuracy-**safe superset** (contains the correct head's blocks + extras), which
is why it passes; MSA's amax subset may drop the correct head's blocks. So the
Stage 2 "mismatch vs Triton" is not proof MSA is wrong — Triton over-selects.

**Candidate fix (if the eval shows it matters):** slice `idx_q` per rank to the
local KV head's own index head → `group=1` → MSA's fixed 16-block budget becomes
the exact intended selection (better than emulating Triton's superset). Flagged
in code (`msa_backend._msa_index_proxy_and_topk`, at the amax reduction and in
the docstring) and in the test (Stage 2 / Stage 2b assert `group==1` exact
parity and a `group>1` subset invariant).

### Ruled out

- **`sparse_topk_select` kernel itself** — Stage 1c passes across
  `num_tiles/total_q/num_heads` and the `num_valid_pages` contract sweep.
- **Index scoring / proxy FMHA** — Stage 1 passes after bug #1.
- **Amax-vs-union as the primary cause** — it does not even trigger unless
  `n_valid_blocks > topk`; both structural bugs (#1, #2) break `group=1` too.

---

## Status

| Bug | Description | Status |
|---|---|---|
| #1 | Global page ids in `_build_kv_indices_and_lens` | Fixed — `677bcb45e5` |
| #2 | Per-query local/init block selection | Fixed — `92d8405af5` |
| #3 | amax-vs-union under-selection (`group>1`, TP artifact) | Open — flagged, pending eval |

Unit stages verified green (SM100): 0, 1a, 1, 1c, 2, 2b. Stages 3 (main
attention dense equivalence) and 4 (full layer) not yet reported.

---

## Performance analysis: the `use_msa=True` block-selection path

The bug-#2 fix restored correctness but made the MSA block-selection step
**less performant**. This section documents the regression, the concrete
optimization opportunities, and the kernel alternatives for recovering the lost
throughput once correctness is locked.

### What regressed

The three commits on the branch:

| Commit | Change | Perf impact |
|---|---|---|
| `677bcb45e5` | remove erroneous page-id clamp in `_build_kv_indices_and_lens` | neutral / slightly positive (one fewer op) |
| `92d8405af5` | replace `fmha_sm100.sparse_topk_select` with torch selection | **regression** (see below) |
| `c35b9673` | add this markdown | none (docs) |

`92d8405af5` swapped **one fused CUDA kernel** (`sparse_topk_select`: a tuned
2-launch transpose + histogram-topk + warp-bitonic-sort pipeline that already
folds in the OOB clamp and the init/local forcing) for a **~15-op eager torch
subgraph plus host-side CPU work** (`_select_blocks_from_maxscore` +
`_per_token_valid_blocks`). This runs **once per sparse layer, every forward
step**, with **CUDA graphs disabled** on this path, so every launch is exposed
on the host critical path.

Concrete costs, in `msa_backend._select_blocks_from_maxscore`:

- `permute(...).to(torch.float32)` materializes a full fp32
  `[total_q, kv, n_blocks]` tensor.
- `.clone()` is **dead work** — every subsequent op (`torch.where`,
  `masked_fill`) is out-of-place and rebinds `scores`, so nothing ever mutates
  it in place. It is a redundant full-tensor copy.
- Two `torch.full_like(scores, ...)` allocate full-size tensors only to feed
  `torch.where`.
- `topk` **plus** a second `torch.sort` (to restore ascending order) — the
  kernel produced ascending order directly.

And in `_per_token_valid_blocks`: CPU tensor construction + a host→device copy
on the exposed path, and — critically — it is **layer-invariant but recomputed
per layer**, as are `_build_kv_indices_and_lens` (`kv_indices`) and
`_qo_lens_offsets_from_metadata`.

For calibration, the MSA kernel's own history notes measured the equivalent
`torch.where + sort + torch.where` post-process at **~84–101 µs/call**, and the
warp-only ascending sort that replaced a `cub::BlockRadixSort` saved **~12
µs/call** — i.e. the torch selection reintroduces exactly the host/GPU cost the
kernel was tuned to remove, multiplied by layer count and by every decode step.

### The 5 optimization opportunities (ordered by impact / effort)

1. **Hoist layer-invariant work out of the per-layer path (biggest, safest
   win).** Compute `n_valid_blocks`, `kv_indices`, and
   `(qo_lens, kv_lens, qo_offset)` **once per forward step** and cache them on
   `MiniMaxM3SparseAttentionMetadata`, then reuse across all sparse layers
   (M3 has ~57 sparse layers, 3..59). These depend only on metadata, not on the
   layer. Removes the redundant CPU work + H2D copies multiplied by layer count,
   which is the dominant new overhead per step.

2. **Delete the redundant `.clone()` and the `torch.full_like` allocations.**
   Drop `scores = scores.clone()` entirely (no in-place mutation follows).
   Replace `torch.where(init_mask, torch.full_like(scores, _INIT_SCORE), scores)`
   with `scores.masked_fill(init_mask, _INIT_SCORE)` (and likewise for
   `_LOCAL_SCORE`). `masked_fill` takes a scalar and avoids two full-tensor
   allocations. Pure, behavior-preserving simplifications.

3. **Consider skipping the fp32 upcast (optional).** `_INIT_SCORE=1e30` /
   `_LOCAL_SCORE=1e29` are ordering sentinels and both fit bf16's range; keeping
   `scores` in the native dtype halves the memory traffic of the whole subgraph.
   Can perturb top-k tie-breaking vs the reference, so gate it behind a parity
   check before shipping.

4. **Re-enable CUDA graphs on the MSA decode path (largest latency win for
   decode).** Decode is latency-bound and this path currently runs eager. It is
   blocked by host syncs / dynamic shapes: `int(qo.sum().item())` in
   `_per_token_valid_blocks`, and `.tolist()` + the Python `for b in
   range(batch)` loop in `_build_kv_indices_and_lens`. Vectorizing those
   (no `.item()` / `.tolist()`, build indices on-device) makes the selection
   subgraph capturable and largely removes the per-step launch overhead.

5. **Fold selection back into a single kernel (see kernel alternatives below).**
   The comment in `_msa_index_proxy_and_topk` already flags "the kernel remains
   available for a future per-query-capable selector." This is the structural
   fix that eliminates the ~15-launch eager subgraph entirely.

Opportunities 1 and 2 are low-risk and recover most of the regression; 4 and 5
are the structural fixes for the remaining exposed launch overhead.

### Why we can't just reshape inputs to the existing scalar kernel

The scalar `sparse_topk_select` applies three effects uniformly across all
`(head, token)` rows:

| Effect | Scalar param | Input-expressible? |
|---|---|---|
| Force init/sink blocks | `force_begin` | already uniform — no problem |
| Force local window | `force_end` (anchored at batch-wide `nvp`) | **yes** — stamp those blocks to a sentinel score per query |
| OOB → `-1` padding | `num_valid_pages` | **no** |

Init forcing is already uniform. Local-window forcing *can* be moved to the
input (pre-write `_LOCAL_SCORE` into each query's `[nvb-local, nvb)` blocks and
pass `force_end=0`). The blocker is the **per-query OOB → `-1` padding**, which
cannot be expressed via inputs:

1. **`-1` has no score preimage.** The kernel emits `-1` solely from a
   comparison on the *output index* (`physical_block_idx >= num_valid_pages`),
   not on any input score. Setting invalid blocks to `-inf` does not help: a
   query with fewer than 16 valid blocks still has its top-16 filled with
   `-inf` blocks emitted at their **real** indices → violates the sparse-FMHA
   contract (`bi < num_pages`) → OOB page reads. That is the original bug.
2. **The index axis can't be per-query remapped.** The emitted index must be the
   true physical block id (the FMHA needs it to fetch KV), and the `K` axis is
   shared across rows, so no per-query compaction makes a single scalar boundary
   correct for everyone.
3. **No single scalar fits the batch.** `nvp = max(nvb)` under-clamps (OOB for
   short queries); `nvp = min(nvb)` over-clamps (drops real blocks for long
   queries). In prefill, every token has its own causal extent, so there is no
   shared value at all.

Consequently any input-only scheme still needs a **per-query post-pass to
convert over-selected tail blocks to `-1`** — which is exactly the
`torch.where + sort + torch.where` work (the bulk of the cost). Reshaping inputs
relocates the cost, it does not remove it. The `-1` clamp is inherently a
per-row, output-index-domain operation, so the kernel is the only place it can
be done cheaply.

### Kernel alternatives (opportunity #5, detailed)

Where the kernel lives (not in TRT-LLM — vendored MSA package):
- FFI entry: `/home/scratch.bbuddharaju_gpu/msa/python/fmha_sm100/csrc/sparse_topk_select.cu`
- Kernel body: `/home/scratch.bbuddharaju_gpu/msa/python/fmha_sm100/csrc/include/sparse_topk_select.cuh`
- Python wrapper: `/home/scratch.bbuddharaju_gpu/msa/python/fmha_sm100/api.py` (`sparse_topk_select`)

The kernel already runs **one row per `(qo_head, token)`**
(`bid = qo_head_idx * total_qo_len + t`), and the transpose store writes a
specific `(q_store, k_store)`. So the per-query bounds can become per-token
arrays with no change to the algorithm — only the parameter types change.

#### Option A — extend `sparse_topk_select` to per-row bounds (recommended)

Turn the batch-wide scalars into per-token device arrays; `force_begin`
(init/sink) is genuinely uniform and stays scalar.

MSA-side changes:
1. `sparse_topk_select.cuh`
   - `SparseTopKTransposeKernel` / `SparseTopKTransposeXorF4Kernel`: change
     `num_valid_pages` and `force_end_start` from `uint32_t` to
     `const uint32_t* __restrict__` (length `total_qo_len`), index by the token
     being written (`q_store` / `q_out`). `is_forced` becomes
     `(k < force_begin) || (k >= fes[q] && k < nvp[q])`.
   - `IndexerTopKWithSortKernel<16>` and `SparseTopKIdentityFillKernel`: change
     `num_valid_pages` to `const uint32_t*` indexed by `t`; clamp becomes
     `idx < num_valid_pages[t]`.
   - `SparseTopKSelect` / `LaunchTransposeAndIndexerTopK`: thread the pointer
     through. Simplest contract: pass `nvp_ptr[total_qo_len]` + scalar
     `force_begin` + scalar `force_end`, and compute
     `fes[q] = nvp[q] >= force_end ? nvp[q] - force_end : 0` in-kernel.
2. `sparse_topk_select.cu` (FFI): change `num_valid_pages` from `int64_t` to a
   `TensorView` (int32, `[total_qo_len]`); add dim/dtype checks.
3. `api.py::sparse_topk_select`: accept `num_valid_pages` as an int (broadcast
   for backward compat) or a `[total_qo_len]` int32 CUDA tensor.

TRT-LLM-side change (`msa_backend.py`): delete `_select_blocks_from_maxscore`,
keep `_per_token_valid_blocks` (materialize it as an int32 CUDA tensor), and
call:

```python
return fmha_sm100.sparse_topk_select(
    max_score_kv.contiguous(),
    _MSA_REQUIRED_TOPK,
    num_valid_pages=n_valid_blocks_cuda,   # now per-token
    force_begin_blocks=init_blocks,
    force_end_blocks=local_blocks,
)
```

This is **behavior-equivalent** to the current torch selection (verified):
init and local are both stamped as the max sentinel (`FLT_MAX` in-kernel vs
`1e30`/`1e29` in torch), but `force_begin + force_end <= topk` is asserted in
`api.py`, so forced blocks never contend for eviction and the sentinel-priority
difference cannot change the result; the reference's `[nvb-local, nvb)` window
equals `[force_end_start, nvp)`; and the reference `>= nvb → -inf` mask equals
the kernel's `>= nvp → -1` clamp. It collapses ~15 launches + an fp32
clone/`full_like` back to **2 launches with no extra memory traffic**, and — if
paired with opportunity #4 (build `n_valid_blocks` on-device without host
syncs) — is CUDA-graph capturable. The `amax` group-reduction (bug #3) stays in
torch and is orthogonal: it neither fixes nor regresses with this change.

Estimated effort: ~30-line kernel change to a fork we already control; net
*removes* code from `msa_backend.py`.

#### Option B — in-tree Triton kernel

Write a `@triton.jit` kernel in TRT-LLM doing masked top-16 + ascending output
per `(token, kv_head)` row, taking per-token `nvb`. Single launch,
graph-capturable, fully owned in-tree, but we take on a new kernel to maintain
and to match against the CUDA reference. Choose this if editing the vendored MSA
package is off the table.

#### Option C — bucket the batch by valid-block count

Issue one scalar `sparse_topk_select` per distinct `nvb` value. Cheap for pure
decode (few distinct sequence-length buckets) but degenerates in prefill (every
token has its own causal extent → one call per token). Decode-only micro-opt,
not a general fix.

**Recommendation:** Option A, paired with opportunities #1 and #4.

---

## Next steps

1. **End-to-end accuracy eval** (the real arbiter of bug #3):
   ```bash
   LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
     pytest "tests/integration/defs/accuracy/test_llm_api_pytorch.py::TestMiniMaxM3MXFP8::test_mxfp8[use_msa=True]" -v -s
   ```
   - If accuracy matches the Triton path → bugs #1+#2 were the whole story;
     decide whether to still do the `group=1` slicing for cleanliness.
   - If still below target → implement bug #3's per-rank index-head slicing and
     re-verify (Stage 2/4 + eval).
2. Run remaining unit stages 3 and 4 on the SM100 box.
3. **Perf follow-up:** the selection now runs in torch (host-side). See
   "Performance analysis" above — land opportunities #1 and #2 (safe wins), then
   Option A + #4 to restore the fused kernel path once correctness is locked.

## How to run the diagnostics

```bash
# Pure CPU / anywhere:
pytest tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_vs_triton_parity.py -k "spec or build_kv_indices" -v

# SM100 + fmha_sm100 (full suite):
pytest tests/unittest/_torch/attention/sparse/test_minimax_m3_msa_vs_triton_parity.py -v -s
```
