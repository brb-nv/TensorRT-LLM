# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""R1A yields folded FP4 ownership to the original P4 consumer."""

import hashlib
import runpy
import tempfile
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
MODELS = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
R1A = runpy.run_path(MODELS / "r1a_runtime.py")


class PrecedenceTests(unittest.TestCase):
    def make_case(self, pending=None, layer=3):
        hidden = NS(
            shape=(176, 6144),
            dtype="bf16",
            device=0,
            is_cuda=True,
            dim=lambda: 2,
            data_ptr=lambda: 123,
            is_contiguous=lambda: True,
        )
        main, aux = object(), object()
        cuda = NS(
            current_stream=lambda: main,
            is_current_stream_capturing=lambda: False,
            stream=lambda _: nullcontext(),
        )
        model = NS(
            torch=NS(bfloat16="bf16", cuda=cuda, compiler=NS(is_compiling=lambda: False)),
            _F3_FP4=NS(pending=pending),
            _F3_STATE={"fp4_layer": layer},
            _SP_LOGITS=NS(pending=None),
            _CTXNORMQ_F32=NS(pending=None),
            MOE_ENGINE_MAX_TOKENS=8,
            S1B_FOLD_CHECK=False,
            AutoTuner=NS(get=lambda: NS(is_tuning_mode=False)),
            flashinfer_mxfp8_autotune_active=lambda: False,
            m31_s1b_fold=NS(_LAYOUT={176: True}),
            ROUTER_TC_GEN=NS(covers=lambda _: False),
            fp4_quantize=Mock(return_value=("quantized", "scales")),
            _decode_route=Mock(return_value=("topk", "routing")),
            _routed_experts=Mock(return_value="experts"),
            logger=NS(info=Mock()),
        )
        scale, target, built = object(), object(), object()
        core = NS(
            mapping=NS(tp_size=2, tp_rank=0),
            hidden=6144,
            w={"embed": NS(device=0)},
            _decode=target,
            _built_for=built,
            aux_stream_moe=aux,
            moe={3: NS(fc31_input_scale=scale)},
            moe_max_num_tokens=224,
            s1b_pins={176: 8},
        )
        state = R1A["_State"].__new__(R1A["_State"])
        state.__dict__.update(
            model=model,
            core=core,
            target=target,
            built_for=built,
            aux=aux,
            scales={3: scale},
            disabled=False,
            thread=threading.get_ident(),
            failed=False,
            active=None,
            events={3: (Mock(), Mock())},
            counts={},
            multi=NS(do_multi_stream=lambda: True, _record_stream=Mock()),
            original_routed=Mock(return_value="stock"),
        )
        state.original_moe = Mock(
            side_effect=lambda i, h, **kw: state.routed(i, h, kw["do_finalize"])
        )
        return state, hidden

    def test_valid_folded_payload_uses_original_consumer_without_aux_quantize(self):
        pending = (123, 176, 6144, "folded", "folded_scales")
        state, hidden = self.make_case(pending)
        self.assertEqual(state.moe(3, hidden, return_parts=False), "stock")
        state.original_routed.assert_called_once_with(3, hidden, True)
        state.model.fp4_quantize.assert_not_called()
        self.assertIs(state.model._F3_FP4.pending, pending)
        self.assertEqual(state.counts[3]["fallback_folded_fp4"], 1)

    def test_no_folded_payload_retains_r1a_aux_quantize_and_consumption(self):
        state, hidden = self.make_case()
        self.assertEqual(state.moe(3, hidden, return_parts=False), "experts")
        state.original_routed.assert_not_called()
        state.model.fp4_quantize.assert_called_once_with(hidden, state.scales[3], 16, False, False)
        state.model._routed_experts.assert_called_once_with(
            state.core, 3, "quantized", "scales", None, True, "routing"
        )

    def test_wrong_layer_pointer_or_extent_stays_with_stock_consumer(self):
        for pending, layer in (
            ((123, 176, 6144, "x", "sf"), 4),
            ((124, 176, 6144, "x", "sf"), 3),
            ((123, 175, 6144, "x", "sf"), 3),
            ((123, 176, 6143, "x", "sf"), 3),
        ):
            state, hidden = self.make_case(pending, layer)
            self.assertEqual(state.moe(3, hidden, return_parts=False), "stock")
            state.model.fp4_quantize.assert_not_called()
            self.assertIs(state.model._F3_FP4.pending, pending)
            self.assertEqual(state.counts[3]["fallback_folded_fp4_miss"], 1)

    def test_model_digest_is_bound_and_a_different_source_is_rejected(self):
        model_path = MODELS / "minimax_m31_nvfp4__sm_107__tp2/modeling.py"
        self.assertEqual(hashlib.md5(model_path.read_bytes()).hexdigest(), R1A["MODEL_MD5"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.py"
            path.write_text("different source\n")
            with self.assertRaisesRegex(RuntimeError, "^R1A requires the pinned TP2 source$"):
                R1A["arm"](NS(__file__=str(path)))


if __name__ == "__main__":
    unittest.main()
