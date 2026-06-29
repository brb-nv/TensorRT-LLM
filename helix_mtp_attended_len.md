# Helix CP + MTP: inactive-rank attended-length bug (first_gen_token)

Scope: **MTP-Eagle one-model speculative decoding** under **Helix context
parallelism (CP)** on the decode/generation side (DeepSeek-V3 / V3-Lite,
trtllm-gen MLA kernels). Disagg setup: context phase = pure TP, decode phase =
Helix CP.

This documents a concrete, root-caused correctness bug found while bringing up
Helix CP + MTP, the experiment that confirmed it, and the proposed real fix.

---

## 1. Symptom

`MTP=1` (`draft_len=1`) fails the `MTP=0` sanity test. On the **first decode
step** after the disagg handoff, for the accepted/first generation token
(`113353`, global position 40) the target/verify forward diverges from `MTP=0`:

- `BEFORE ATTENTION` hidden states **match** `MTP=0` (embedding/residual fine).
- `AFTER ATTENTION` hidden states **diverge** (e.g. `[0.0094, -0.0278, …]`
  instead of `MTP=0`'s `[0.0245, 0.0007, …]`).
- End-to-end the wrong token is produced (`vian arms` instead of `vian medium`).

So the divergence is created *inside* MLA attention for `token0`, even though
its input is identical to the single-token (`MTP=0`) case.

---

## 2. Root cause

The whole bug is a mismatch between the per-query-row KV length the Helix
semantics *require* and the one the trtllm-gen MLA-gen kernel *assumes*. State
both with a single quantity and the mismatch becomes obvious.

### 2.1 The correct per-row rule

For a query block of length `q_len` (one verify/decode step of a sequence),
index the query rows `r = 0 … q_len-1`. Per CP rank, let:

- `cached` = KV already in **this rank's** cache before this step, and
- `owned_count(r)` = number of query rows `j ≤ r` whose KV **this rank actually
  wrote** this step.

Under Helix the new tokens' KV is written to **exactly one** CP rank
(`helixKvWriteSlot` in `mlaKernels.cu` returns `-1` on ranks that don't own a
token), and the locally-owned new tokens are appended **contiguously at the KV
tail in row order**. So the exact correct effective KV length for row `r`, on
*any* rank, is:

```
effective_kv(r) = cached + owned_count(r)
```

A slope-free read of `kv[0 : effective_kv(r)]` lands on exactly `[0, cached)`
(all strictly in the past, always attended) **plus** precisely the owned
new-token slots with `j ≤ r` (causal, and physically present on this rank). This
single formula is correct for every rank and every ownership pattern.

`owned_count(r)` is just a within-sequence cumulative sum of
`~helix_is_inactive_rank` (the per-query-token ownership flags already plumbed
through `TrtllmAttentionMetadata`; see `model_engine.py` `_helix_verify_token_params`,
which assigns ownership via `(decode_index // tokens_per_block) % cp_size`).

### 2.2 What the kernel does instead (and why it's wrong)

On the trtllm-gen kernel for linear MTP, `is_spec_decoding_enabled` is forced
`False` (see `TrtllmAttentionMetadata.update_spec_dec_param`), so there is **no
explicit packed mask**. The linear path (`fmhaKernels.h`, `mIsCausalSpecDecodingGen`)
instead derives the per-row bound from a **single per-request scalar**
`seqLensKv` (= `kv_cache_lengths`, the prefix-sum `seqKVOffsets` in
`mlaKernels.cu`):

```
effective_kv_kernel(r) = seqLensKv - (q_len - 1 - r)
```

This hardcodes the assumption that **the owned new tokens are the bottom-aligned,
contiguous last `owned_count` rows of the block**. That assumption — owned rows ==
the contiguous bottom rows — is the entire bug surface:

- It is satisfied by an **owns-all** rank (`seqLensKv = cached + q_len`) and by a
  rank that owns a **contiguous tail** of rows.
- It is violated by an **inactive** rank (owns none — there is no tail, yet the
  kernel still reserves `q_len-1` tail slots and steals real cached positions)
  and by a rank that owns a **lower prefix** of rows (the reservation lands on the
  wrong rows entirely).

A single per-sequence scalar plus the intrinsic `+1`-per-row slope simply cannot
express `owned_count(r)` when the owned rows are not a bottom tail.

### 2.3 Worked examples

All examples use the §1 first-decode scenario: prompt len 40, `cp_size=2`,
block-cyclic split → active rank `cached=32`, inactive rank `cached=8`.

**Example 1 — inactive rank, MTP=0 (`q_len=1`), owns nothing.** `seqLensKv = 8`.

| row | kernel `seqLensKv-(q_len-1-r)` | correct `cached + owned_count(r)` |
|-----|-------------------------------|------------------------------------|
| 0   | `8 - 0 = 8`                   | `8 + 0 = 8`                        |

Match — but **correct only by luck**: with one row there is no tail to mis-reserve.

**Example 2 — inactive rank, MTP=1 (`q_len=2`), owns nothing.** `seqLensKv = 8`.

| row | kernel              | correct      |
|-----|---------------------|--------------|
| 0   | `8 - 1 = 7` ✗ drops a real cached token | `8 + 0 = 8` |
| 1   | `8 - 0 = 8`         | `8 + 0 = 8`  |

Row 0 silently loses one cached position — exactly the "Σexp shrinks on row 0"
softmax-stats signature in §3.2. The rule repairs it.

**Example 3 — active rank, MTP=1 (`q_len=2`), owns both tokens.**
`seqLensKv = cached + 2 = 34`.

| row | kernel        | correct `32 + owned_count(r)` |
|-----|---------------|--------------------------------|
| 0   | `34 - 1 = 33` | `32 + 1 = 33`                  |
| 1   | `34 - 0 = 34` | `32 + 2 = 34`                  |

**Identical** — the rule is a strict generalization, so the (already correct)
active rank is unchanged. This matches the "bit-identical active rank" finding in
§3.2.

**Example 4 — the straddle (mixed ownership), MTP=2 (`q_len=3`).** Take
`tokens_per_block = 4` and draft decode-indices `3,4,5`; owners are
`(idx // 4) % 2`: index 3 → rank0, indices 4,5 → rank1. So **rank0 owns only row 0**
and **rank1 owns rows 1,2**.

rank0 (owns the *lower* row; `owned_count = 1`; `seqLensKv = cached0 + 1`):

| row | kernel `(cached0+1)-(2-r)` | correct `cached0 + owned_count(r)` |
|-----|---------------------------|-------------------------------------|
| 0   | `cached0 - 1` ✗           | `cached0 + 1`                       |
| 1   | `cached0` ✗               | `cached0 + 1`                       |
| 2   | `cached0 + 1` ✗           | `cached0 + 1`                       |

**Every row wrong** — the owned token is row 0, but the kernel reserves the tail
for it.

rank1 (owns the *upper* rows; `owned_count = 2`; `seqLensKv = cached1 + 2`):

| row | kernel `(cached1+2)-(2-r)` | correct `cached1 + owned_count(r)` |
|-----|---------------------------|-------------------------------------|
| 0   | `cached1` ✓               | `cached1 + 0 = cached1`             |
| 1   | `cached1 + 1` ✓           | `cached1 + 1`                       |
| 2   | `cached1 + 2` ✓           | `cached1 + 2`                       |

**Correct — but only by coincidence**, because here the owned rows happen to be
the bottom-aligned tail.

### 2.4 Summary

The scalar+slope kernel is correct **iff** the rows a rank owns are the contiguous
bottom rows of the block (owns-all, or owns-a-tail). It fails for **owns-none**
(inactive) and **owns-a-lower-prefix** (straddle) ranks — and a verify block
straddling a `tokens_per_block` boundary genuinely produces the lower-owning case,
so the defect is **per-row, not merely a "first token" special case**. Severity
scales with draft length: every owned row whose `owned_count(r)` differs from the
tail-aligned assumption is corrupted. The corrupted partial then flows through the
Helix softmax/LSE combine (`_helix_post_process`), producing a wrong combined
output. The fix is to make each row attend `cached + owned_count(r)` KV positions
(see §4–§5).

---

## 3. How we confirmed it

Three independent pieces of evidence, all on the first decode step, `token0`:

### 3.1 KV-length metadata (host, `TrtllmAttentionMetadata.prepare`)

Per-rank trace (`cp_size=2`, prompt len 40 split block-cyclically 32/8):

| run   | rank      | seq_lens_kv (q_len) | cached | final kv_len |
|-------|-----------|---------------------|--------|--------------|
| MTP=0 | active    | 1 | 32 | 33 |
| MTP=0 | inactive  | 1 |  8 |  8 |
| MTP=1 | active    | 2 | 32 | 34 |
| MTP=1 | inactive  | 2 |  8 |  8 |

The inactive rank's `kv_len` stays 8 while `q_len` grows 1→2 — exactly the
condition that shrinks token0's window.

### 3.2 Softmax stats (measured kernel output, `_attn_forward_gen` pre-combine)

The attention kernel writes `softmax_stats = [running_max, Σexp]` per head. On the
inactive rank, holding `cached = 8` fixed and only changing `q_len` 1→2, token0's
denominators shrink and one head's max drops — the signature of removing exactly
one KV position:

| head | MTP=0 `[m, l]` | MTP=1 `[m, l]` | interpretation |
|------|----------------|----------------|----------------|
| 0 | `[1.1545, 1.9186]` | `[1.1545, 1.8684]` | max same, Σexp ↓ → dropped a non-max key |
| 3 | `[6.4279, 1.7289]` | `[6.4279, 1.3271]` | same signature |
| 5 | `[1.7136, 3.0572]` | `[1.0510, 3.9909]` | max dropped → the removed key was this head's argmax |

On the **active** rank token0's `softmax_stats` and `partial_o` are
**bit-identical** between MTP=0 and MTP=1 (its window is `cached + 1`
regardless of `q_len`), confirming the defect is isolated to the inactive rank.

### 3.3 Direct fix probe (the decisive test)

We bumped the inactive rank's `kv_len` by `q_len - 1` (8 → 9) in
`prepare()`, so the tail reservation lands past the real cached tokens and
`token0` (row 0) again attends the full `cached` prefix. This is memory-safe for
token0 (it reads at most slot `cached-1`); only later rows read into the unused,
already-allocated tail of the KV page.

Result (MTP=1 with the bump), `token0`, inactive rank — **every checkpoint
snapped back to the MTP=0 baseline**:

- `softmax_stats` row0 → identical to MTP=0
- `partial_o` token0 → identical to MTP=0
- `AFTER ATTENTION` token0 → `[0.0245, 0.0007, 0.0232, …]` (= MTP=0)
- token0 logits → `[-0.4082, -1.6875, -0.4258, …]` (= MTP=0)
- **end-to-end output → `vian medium`** (= MTP=0; broken run produced `vian arms`)

This closes the loop: the inactive-rank causal-tail reservation is the **entire**
root cause for the first_gen_token.

> The probe was a **diagnosis-only** hack and has been reverted. It repairs only
> row 0 (with `q_len - 1` bump the slope is unchanged, so row `r` attends
> `cached + r`, over-reading for `r ≥ 1`). It is sufficient for `draft_len=1`
> (where the accepted token is driven entirely by row 0) but is **not** a general
> fix.

---

## 4. Proposed fix

On an inactive rank, **every** query row must attend the full cached prefix
`[0, cached)` with **no causal tail slope** (slope 0, not +1) — the query tokens'
KV is not local, so there is nothing to mask among them. A single per-sequence
`kv_len` scalar cannot express this because the +1-per-row slope is applied inside
the FMHA.

Preferred direction:

1. **Plumb `helix_is_inactive_rank` (per-token) into the MLA generation attention
   mask.** It already reaches the kernel boundary for KV writes
   (`helixKvWriteSlot`); it just needs to reach the masking too. For query rows
   flagged inactive, disable the causal tail so they attend all `kv_cache_len`
   cached positions. This is the principled fix and matches the existing
   ownership plumbing.

Alternative considered:

2. Enable a custom packed spec-dec mask for the inactive-rank case. Today
   `is_spec_decoding_enabled` is forced `False` on the trtllm-gen kernel for
   linear MTP, so there is no mask channel at all; enabling one is a larger change
   than (1).

### Next steps

1. Trace the trtllm-gen MLA generation FMHA to locate exactly where the causal
   bound is derived, and confirm what per-query signal it can consume. **(done —
   see §5 below)**
2. Implement the chosen approach (see §5): make inactive-rank query rows attend
   the full cached prefix.
3. Validate on `draft_len = 1` **and** `draft_len ≥ 2` (the bug worsens with
   draft length — verify all rows on the inactive rank attend the full cached
   prefix, not just row 0).
4. Re-run the MTP=0 vs MTP=1 sanity comparison; `AFTER ATTENTION`, logits, and the
   end-to-end output for `token0` must match MTP=0.

---

## 5. Fix scoping (trtllm-gen MLA generation path)

### 5.1 How the MLA-gen kernel derives its causal bound

The MLA generation attention runs through `AttentionOp::mlaGeneration`
(`attentionOp.cpp`) using the trtllm-gen runner, with:

```cpp
// attentionOp.cpp ~1085
// MLA generation kernels use dense mask. For multi-token generation, TRTLLM-Gen
// applies causality by shrinking each token's effective KV length.
tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Dense;
tllmRunnerParams.seqLensKvPtr = params.cache_seq_lens;   // one value per request
tllmRunnerParams.mMaxSeqLenQ  = params.acc_q_len / batch_beam;
```

In `fmhaKernels.h`:

- `algoFilterForCubinPath` (~296) rewrites `Dense → Causal` for generation
  kernels **but explicitly skips MLA-gen** (`!options.mIsMlaGen`), so MLA-gen
  stays `Dense`.
- Multi-token mode is then split (`~1013`):
  ```cpp
  options.mIsCustomSpecDecodingGen = !isContext && mMaxSeqLenQ > 1 &&  mIsSpecDecTree; // tree → custom mask
  options.mIsCausalSpecDecodingGen = !isContext && mMaxSeqLenQ > 1 && !mIsSpecDecTree; // linear → implicit shrink
  ```

So **linear MTP (`mIsSpecDecTree = false`) lands in `mIsCausalSpecDecodingGen`** —
the cubin applies the per-query-row causal shrink
`effective_kv(r) = seqLensKv - (mMaxSeqLenQ - 1 - r)` internally, from a single
per-request `seqLensKv`. That is exactly the bug: there is no per-query-row KV
input, only a per-request scalar plus the intrinsic `+1`-per-row slope.

### 5.2 The kernel is helix-unaware on this path

`helix_is_inactive_rank` is set on `xqaParams` (`attentionOp.cpp:295`) and on the
RoPE/KV-write `preprocessingParams` (`:1754`), but **not** on the trtllm-gen
`tllmRunnerParams` inside `mlaGeneration`. So the MLA-gen attention kernel has no
knowledge of helix inactivity; it cannot currently behave differently for
inactive ranks.

### 5.3 Candidate fixes

**Approach A — route inactive-rank verify through the Custom-mask gen path
(preferred, kernel-aware, general).**
The same MLA-gen kernel already supports a `Custom` mask
(`mIsCustomSpecDecodingGen`, `runPrepareCustomMask`, `customMaskPtr`) for tree
spec-dec, so the cubins exist. Build a per-`(query_row × kv)` mask that encodes
the true helix semantics:
- query rows this rank **owns**: causal as today;
- query rows this rank does **not** own (inactive): unmasked over the full cached
  prefix (and over no self-tokens, since none were written).

Requires: plumb `helix_is_inactive_rank` into `mlaGeneration`/`tllmRunnerParams`,
enable the custom-mask path for the helix case (currently gated on
`mIsSpecDecTree`), and add a mask builder. Handles every case — `draft_len ≥ 2`
and even mixed ownership when a q-block straddles a `tokens_per_block` boundary.

**Approach B — decompose inactive-rank attention (no C++/cubin change).**
On fully-inactive ranks, run the `q_len` query tokens as `q_len` independent
`q_len = 1` decodes (each `kv = cached`), then assemble per-token `partial_o` +
`softmax_stats` for the Helix combine. Correct for all rows. Costs `q_len` extra
launches on inactive ranks and needs care with the combine layout and CUDA-graph
capture. Contained entirely in the Python MLA/Helix path.

**Approach C — kernel flag to skip the causal shrink for inactive ranks
(most direct, least feasible here).**
Pass `helix_is_inactive_rank` to `tllmRunnerParams` and have the MLA-gen kernel
use the full per-request `seqLensKv` for every query row when inactive. Cleanest
semantically, but the trtllm-gen MLA-gen kernels are prebuilt/exported cubins;
adding a non-causal multi-token mode likely requires regenerating them, which is
out of scope for this repo.

### 5.4 Recommendation

**Approach A.** It reuses the existing Custom-mask MLA-gen machinery (already
exercised by tree spec-dec on the same kernel), expresses the exact per-query
semantics Helix needs, and generalizes to longer draft lengths and
mixed-ownership q-blocks. Approach B is a reasonable fallback if enabling the
custom-mask path for non-tree generation proves too invasive.

> **Superseded — see §6 for the direction we actually chose.** Approach A turned
> out to require editing trtllm-gen host/kernel-selection code (un-gating the
> custom-mask path in `fmhaKernels.h`, plumbing `tllmRunnerParams`, a C++ mask
> builder). **trtllm-gen kernel changes are a no-go** for this work, which also
> rules out Approach C (cubin regeneration). That leaves a Python-only refinement
> of Approach B as the chosen direction.

### 5.5 Open questions to resolve before implementing

1. Confirm the `Custom`-mask MLA-gen cubins are selected/available for the target
   SM (they should be, since dynamic-tree DeepSeek uses MLA-gen + custom mask).
2. Per-query ownership for mixed q-blocks: confirm the per-token
   `helix_is_inactive_rank` flags (and `_helix_owned_token_counts`) give the
   per-row ownership the mask builder needs when a block straddles a
   `tokens_per_block` boundary.
3. CUDA-graph compatibility of the chosen approach (mask buffers must be static /
   capturable; Approach B's variable launch count is graph-hostile).
4. **The MTP layer needs the same fix** — the second `[mlaGenKernel]` block in the
   trace (after `SpecDecOneEngineForCausalLM`) shows the identical inactive-rank
   shrink, so whatever path is chosen must cover both the target verify and the
   MTP-layer attention.

---

## 6. Final direction (chosen): flatten helix verify into per-row single-token decodes

Constraint: **no trtllm-gen kernel changes** (rules out Approaches A and C). The
chosen fix is a Python-only generalization of Approach B that realizes the §2.1
rule directly, in a single kernel launch, for *every* ownership pattern.

### 6.1 The idea

The entire defect (§2.2) is the kernel's intrinsic `+1`-per-row causal slope,
applied from a single per-request scalar `seqLensKv` while in the
`mIsCausalSpecDecodingGen` multi-token mode (`mMaxSeqLenQ > 1`). So **don't use
that mode** for the helix generation attention: present each query row as its own
`q_len = 1` decode "request" (`mMaxSeqLenQ = 1`), and give each row the exact
§2.1 bound as its per-request scalar:

```
seqLensKv(r) = cached + owned_count(r)
```

With `mMaxSeqLenQ = 1` the slope term `(mMaxSeqLenQ - 1 - r)` is identically `0`,
so the per-request scalar *is* the per-row bound. The kernel reads
`kv[0 : cached + owned_count(r)]`, which is exactly the slope-free read §2.1
prescribes: the full cached prefix `[0, cached)` plus precisely the owned
new-token slots with `j ≤ r`. No slope means no tail mis-reservation, and trtllm-gen
needs no change (the path simply doesn't enter `mIsCausalSpecDecodingGen`, which
is gated on `mMaxSeqLenQ > 1`; `is_spec_decoding_enabled` is already forced
`False` here anyway).

### 6.2 Why this is correct for *all* ranks (not just inactive)

`owned_count(r)` is the within-sequence prefix-sum of `~helix_is_inactive_rank`:

- **inactive rank** → `cached + 0` for every row (fixes Examples 1–2).
- **active owns-all** → `cached + (r + 1)`, **bit-identical** to today's correct
  active-rank result `(cached + q_len) - (q_len - 1 - r)` (Example 3) — strict
  generalization, active rank unchanged.
- **straddle / lower-prefix** → also exactly correct (fixes Example 4, which
  Approach-B-as-originally-written, scoped to "fully-inactive ranks only", does
  **not** cover).

### 6.3 Why it beats B-as-written and resolves the §5.5 open questions

- **CUDA-graph friendly (§5.5.3):** the flattened row count is
  `num_seqs × (draft_len + 1)` — static for a captured graph, unlike B's
  data-dependent variable launch count. One launch, not `q_len` launches.
- **Combine is already per-row:** `_helix_post_process` consumes
  `partial_o [num_tokens, …]` and `softmax_stats [num_tokens, num_heads, 2]` —
  exactly the per-row outputs the flattened batch produces. No combine rework.
- **Covers the MTP layer (§5.5.4):** both the target-verify and the MTP-layer
  `[mlaGenKernel]` blocks run through the same helix generation path, so
  flattening fixes both.
- **Per-row ownership (§5.5.2):** comes straight from the per-token
  `helix_is_inactive_rank` flags via the segmented prefix-sum, including the
  straddle case.

### 6.4 Implementation plan (all in the Python MLA/Helix path)

1. In `TrtllmAttentionMetadata.prepare()` (helix + spec-dec branch), build a
   **per-query-row** kv-length vector `cached + cumsum(~helix_is_inactive_rank)`
   segmented by sequence, instead of the current per-sequence
   `cached + owned_total`.
2. Reshape the helix generation attention call so each query row is a `q_len = 1`
   request: `mMaxSeqLenQ = 1`, per-row `seqLensKv`, and KV block offsets
   replicated per row of the same sequence (they share the same physical blocks).
3. Keep the RoPE + KV-write step (`helixKvWriteSlot`,
   `applyMLARopeAndAssignQKVKernelGeneration`) on the un-flattened owned-token
   representation so owned tokens still land contiguously at the KV tail in row
   order — the flattened read side then sees exactly the owned tokens `j ≤ r`.
4. Remove the temporary debug `print`s in `prepare()` / `_attn_forward_gen()`.

### 6.5 Validation

- `draft_len = 1` **and** `draft_len ≥ 2`, plus a deliberate
  `tokens_per_block`-straddle case to exercise the lower-prefix path.
- Per-row `softmax_stats` / `partial_o` / `AFTER ATTENTION` / logits / end-to-end
  token must match `MTP=0`; active-rank rows must stay bit-identical.

### 6.6 Implementation notes (as landed)

Implemented purely on the Python read path; no C++ / cubin changes.

- `tensorrt_llm/_torch/attention_backend/trtllm.py`
  - `_maybe_prepare_helix_flatten()` (called from `prepare()`): builds the
    flattened per-query-row tensors into CUDA-graph-static buffers. The per-row
    KV bound is `repeat_interleave(cached, q_len) + owned_count(r)`, where
    `owned_count(r)` is the within-sequence inclusive prefix-sum of
    `~helix_is_inactive_rank`. Gated on `enable_helix and num_contexts == 0 and
    helix_is_inactive_rank_cpu is not None and total_q > num_seqs`. Note it does
    **not** gate on `is_spec_decoding_enabled`: `update_spec_dec_param` forces
    that False on the trtllm-gen kernel for linear MTP, so the verify flows
    through the per-sequence kv_lens branch. The robust discriminator for "a
    multi-token verify exposed to the slope bug" is simply `total_q > num_seqs`
    (more query rows than sequences); plain `q_len == 1` decode is left untouched
    since its slope term is already zero. **Ordering matters:** the call runs
    *after* `copy_batch_block_offsets()` so the per-row block-table replication
    reads this step's `kv_cache_block_offsets`, not a stale/zero buffer (reading
    it too early produced all-zero K/V → `softmax_stats = [0, kv_len]` and
    `partial_o = 0`).
  - `helix_flattened_generation()` context manager: swaps the per-request
    runtime fields the trtllm-gen MLA-gen FMHA consumes — `kv_lens_*_runtime`,
    `prompt_lens_*_runtime` (this sets C++ `num_seqs =
    host_context_lengths.size(0)`, hence `mMaxSeqLenQ == 1`),
    `host_request_types_runtime`, `host_total_kv_lens`, `kv_cache_block_offsets`
    (block table replicated per query row), and `max_num_requests` — then
    restores them. Yields flattened `cu_q_seqlens` (`[0..total_q]`) and
    `cu_kv_seqlens` (exclusive prefix sum of the per-row KV lengths), both of
    which the C++ op size-checks against `num_seqs + 1`.
- `tensorrt_llm/_torch/modules/attention.py` `_attn_forward_gen()`: wraps the
  generation `attn_backend.forward` in `helix_flattened_generation()` and routes
  the flattened `cu_q_seqlens` / `cu_kv_seqlens` through `kwargs`. The KV write
  (`mla_rope_generation`) runs earlier and is unaffected.
- Removed the temporary debug prints in `prepare()` and `_attn_forward_gen()`.
  The model-level `BEFORE/AFTER ATTENTION` etc. prints are left in place for the
  §6.5 validation pass.

**Caveat to verify on hardware:** the flatten presents `total_q` generation
"requests", so it bumps `max_num_requests` for the call. The C++ op sizes its
semaphore / workspace arrays from `max_num_requests`; the eager warmup forward
must therefore exercise this path at the padded batch so the workspace is large
enough *before* CUDA-graph capture (capture cannot `resize_` the workspace). If
graph capture complains about workspace size, pre-size `effective_workspace`
for `max_num_tokens` requests.

---

## 5. Key code references

| Concern | Location |
|---|---|
| Helix kv_len computation (active vs inactive) | `tensorrt_llm/_torch/attention_backend/trtllm.py` `TrtllmAttentionMetadata.prepare()` |
| Per-rank owned-token count | `trtllm.py` `_helix_owned_token_counts()` |
| `is_spec_decoding_enabled` forced False on trtllm-gen linear MTP | `trtllm.py` `update_spec_dec_param()` |
| Helix KV write-slot (`-1` on inactive ranks) | `cpp/tensorrt_llm/kernels/mlaKernels.cu` `helixKvWriteSlot()` |
| Generation RoPE + KV write kernel | `mlaKernels.cu` `applyMLARopeAndAssignQKVKernelGeneration` |
| `seqKVOffsets` (kv extent = prefix sum of kv_cache_lengths) | `mlaKernels.cu` block-scan section |
| Helix attention + LSE combine | `tensorrt_llm/_torch/modules/attention.py` `_attn_forward_gen` / `_helix_post_process` |
