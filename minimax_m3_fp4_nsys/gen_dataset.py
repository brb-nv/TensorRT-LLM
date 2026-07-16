#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a tokenized trtllm-bench dataset with a fixed ISL and OSL.

Each JSONL line is::

    {"task_id": <i>, "input_ids": [<isl random ids>], "output_tokens": <osl>}

which is the schema ``trtllm-bench ... throughput --dataset`` consumes directly
(the engine runs with skip_tokenizer_init, so pre-tokenized ``input_ids`` are
used verbatim -- no tokenizer round-trip, and the exact ISL is guaranteed).

Prompts are distinct random token IDs (fixed seed for reproducibility). Distinct
prompts are used because the profiling workload disables KV-cache block reuse;
even so, distinct prompts keep every request's prefill genuine.

Random IDs are sampled from ``[low, vocab_size)``. ``vocab_size`` is read from
the model's ``config.json`` (searched recursively so nested text/language
configs are handled); if it cannot be found a conservative fallback is used
that is safe for any large-vocab model.
"""

import argparse
import json
import os
import random
import sys

# Conservative upper bound used only if vocab_size cannot be read from the
# model config. MiniMax-M3 (and every other large LLM here) has a far larger
# vocab, so IDs in [low, FALLBACK_VOCAB) are always valid.
FALLBACK_VOCAB = 30000
# Skip the lowest IDs, which are usually reserved special tokens.
LOW_ID = 16


def _find_vocab_size(obj) -> int | None:
    """Recursively search a parsed config dict for the first ``vocab_size``."""
    if isinstance(obj, dict):
        if isinstance(obj.get("vocab_size"), int):
            return obj["vocab_size"]
        for value in obj.values():
            found = _find_vocab_size(value)
            if found is not None:
                return found
    return None


def _resolve_vocab_size(model_dir: str) -> int:
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path) as fh:
            config = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[gen_dataset] WARNING: could not read {config_path} ({exc}); "
              f"falling back to vocab_size={FALLBACK_VOCAB}.", file=sys.stderr)
        return FALLBACK_VOCAB
    vocab = _find_vocab_size(config)
    if vocab is None or vocab <= LOW_ID + 1:
        print(f"[gen_dataset] WARNING: no usable vocab_size in {config_path}; "
              f"falling back to vocab_size={FALLBACK_VOCAB}.", file=sys.stderr)
        return FALLBACK_VOCAB
    return vocab


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Local HF model dir (used to read vocab_size).")
    parser.add_argument("--isl", type=int, default=8192,
                        help="Input sequence length (tokens per prompt).")
    parser.add_argument("--osl", type=int, default=1,
                        help="Output sequence length (output_tokens).")
    parser.add_argument("--num-requests", type=int, default=64,
                        help="Number of dataset rows to emit.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible prompts.")
    args = parser.parse_args()

    vocab_size = _resolve_vocab_size(args.model)
    hi = vocab_size
    assert hi > LOW_ID, f"vocab_size={hi} too small to sample prompts"

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        for task_id in range(args.num_requests):
            input_ids = [rng.randrange(LOW_ID, hi) for _ in range(args.isl)]
            fh.write(json.dumps({
                "task_id": task_id,
                "input_ids": input_ids,
                "output_tokens": args.osl,
            }) + "\n")

    print(f"[gen_dataset] wrote {args.num_requests} rows "
          f"(isl={args.isl}, osl={args.osl}, vocab<{hi}) -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
