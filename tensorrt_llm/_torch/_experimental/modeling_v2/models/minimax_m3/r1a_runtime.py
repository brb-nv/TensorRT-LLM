# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R1A aux-Q scheduling on the stock TP2 MoE; no evidence/factory wrappers."""

import functools
import hashlib
import threading
import types
from collections import Counter
from pathlib import Path

VERSION = "r1a-p4-brb-v1"
MODEL_MD5 = "64009206b8f1bbc9d9c35ff3947e27d7"


def _folded_fp4_reason(model, layer, hidden):
    pending = model._F3_FP4.pending
    if pending is None:
        return None
    # Leave consumption and clearing to the original P4 routed consumer.
    if (
        model._F3_STATE.get("fp4_layer") == layer
        and hidden.dim() == 2
        and hidden.data_ptr() == pending[0]
        and tuple(hidden.shape) == (pending[1], pending[2])
    ):
        return "folded_fp4"
    return "folded_fp4_miss"


class _State:
    def __init__(self, model, core, multi, disabled):
        self.model, self.core, self.multi = model, core, multi
        self.target = core._decode
        self.disabled = disabled
        self.thread = threading.get_ident()
        self.built_for = core._built_for
        self.aux = core.aux_stream_moe
        self.scales = {i: core.moe[i].fc31_input_scale for i in core.moe_layers}
        self.original_moe = self.target._moe
        self.original_routed = self.target._routed
        self.active = None
        self.failed = False
        self.counts = {i: Counter() for i in sorted(core.moe_layers)}
        self.events = {}
        # Events are created before graph capture; the killed arm creates none.
        if not disabled:
            with model.torch.cuda.device(core.w["embed"].device):
                self.events = {
                    i: (model.torch.cuda.Event(), model.torch.cuda.Event()) for i in core.moe_layers
                }

    def check_thread(self, allow_failed=False):
        if threading.get_ident() != self.thread:
            raise RuntimeError("R1a target used from another host thread")
        if self.failed and not allow_failed:
            raise RuntimeError("R1a target poisoned by an earlier failed invocation")

    def eligibility(self, i, h, push):
        model, core, torch = self.model, self.core, self.model.torch
        if self.disabled:
            return "kill"
        if i not in self.events or i not in range(3, 60):
            return "layer"
        if core.mapping.tp_size != 2 or core.hidden != 6144:
            return "model_domain"
        if h.dim() != 2 or tuple(h.shape) != (176, 6144):
            return "shape"
        if h.dtype != torch.bfloat16 or not h.is_cuda or not h.is_contiguous():
            return "layout"
        if h.device != core.w["embed"].device:
            return "device"
        if core._decode is not self.target or core._built_for != self.built_for:
            raise RuntimeError("R1a target/weight lifecycle changed without reinstall")
        if (
            core.aux_stream_moe is not self.aux
            or core.moe[i].fc31_input_scale is not self.scales[i]
        ):
            raise RuntimeError("R1a stream/scale ownership changed")
        if push is not None or core.moe_max_num_tokens < 176:
            return "push_or_chunk"
        if (
            model.MOE_ENGINE_MAX_TOKENS >= 176
            or model.S1B_FOLD_CHECK
            or torch.compiler.is_compiling()
        ):
            return "engine_or_reference_or_compile"
        if model.AutoTuner.get().is_tuning_mode or model.flashinfer_mxfp8_autotune_active():
            return "tuning"
        if not self.multi.do_multi_stream() or self.aux is None:
            return "serial"
        if self.aux == torch.cuda.current_stream():
            return "same_stream"
        if (
            not core.s1b_pins
            or core.s1b_pins.get(176) != 8
            or model.m31_s1b_fold._LAYOUT.get(176) is not True
            or model.ROUTER_TC_GEN.covers(176)
        ):
            return "route_domain"
        # Do not destructively take another producer's slot to test eligibility.
        if model._SP_LOGITS.pending is not None or model._CTXNORMQ_F32.pending is not None:
            return "alternate_router_producer"
        return _folded_fp4_reason(model, i, h)

    def moe(self, i, h, *, return_parts, do_finalize=True, push=None):
        self.check_thread()
        if self.active is not None:
            raise RuntimeError("R1a nested invocation on one target")
        reason = self.eligibility(i, h, push)
        cuda = self.model.torch.cuda
        capturing = (
            not self.model.torch.compiler.is_compiling() and cuda.is_current_stream_capturing()
        )
        phase = "capture" if capturing else "eager"
        counts = self.counts.setdefault(i, Counter())
        counts[phase + "_calls"] += 1
        counts[phase + "_rows_" + str(h.shape[0] if h.dim() == 2 else 0)] += 1
        if reason is not None:
            counts["fallback_" + reason] += 1
        self.active = dict(layer=i, hidden=h, used=False, payload=None, candidate=reason is None)
        main = cuda.current_stream() if reason is None else None
        try:
            if reason is None:
                ready, done = self.events[i]
                ready.record()
                with cuda.stream(self.aux):
                    ready.wait()
                    self.active["payload"] = self.model.fp4_quantize(
                        h, self.scales[i], 16, False, False
                    )
                    done.record()
                counts[phase + "_taken"] += 1
                if counts[phase + "_taken"] == 1:
                    self.model.logger.info(
                        f"[R1A_TAKEN] rank={self.core.mapping.tp_rank} layer={i} "
                        f"rows=176 phase={phase} do_finalize={do_finalize}"
                    )
            result = self.original_moe(
                i, h, return_parts=return_parts, do_finalize=do_finalize, push=push
            )
            if reason is None and not self.active["used"]:
                raise RuntimeError("R1a quantized payload was not consumed")
            if reason is None:
                payload = self.active["payload"]
                if not capturing:
                    self.multi._record_stream(payload, main)
                    self.multi._record_stream(h, self.aux)
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = None

    def routed(self, i, h, do_finalize):
        active = self.active
        if active is None or not active["candidate"]:
            return self.original_routed(i, h, do_finalize)
        if i != active["layer"] or h is not active["hidden"] or active["used"]:
            raise RuntimeError("R1a stale/wrong-layer/double-consumed invocation")
        active["used"] = True
        kind, topk = self.model._decode_route(self.core, i, h)
        if kind != "topk":
            raise RuntimeError("R1a eligible route changed after quant prequeue")
        self.events[i][1].wait()
        x, sf = active["payload"]
        return self.model._routed_experts(self.core, i, x, sf, None, do_finalize, topk)


def _install(model, core, multi, disabled):
    old = core.__dict__.get("_r1a_state")
    if old is not None and old.target is core._decode:
        old.check_thread()
        if any(getattr(old.target, name).__func__ is not fn for name, fn in old.bindings.items()):
            raise RuntimeError("R1a installed methods were replaced")
        if old.built_for != core._built_for:
            raise RuntimeError("R1a same target changed weight identity")
        return old
    if old is not None:
        raise RuntimeError("restart worker before replacing the R1A target")
    if (
        core._decode._moe.__func__ is not model.DecodeTarget._moe
        or core._decode._routed.__func__ is not model.DecodeTarget._routed
    ):
        raise RuntimeError("R1a cannot compose with unknown target wrappers")
    state = _State(model, core, multi, disabled)
    core._r1a_state = state

    # Bound closures preserve exact public call signatures and original return forms.
    def moe(_target, i, h, *, return_parts, do_finalize=True, push=None):
        return state.moe(i, h, return_parts=return_parts, do_finalize=do_finalize, push=push)

    def routed(_target, i, h, do_finalize):
        return state.routed(i, h, do_finalize)

    state.target._moe = types.MethodType(moe, state.target)
    state.target._routed = types.MethodType(routed, state.target)
    state.bindings = {"_moe": moe, "_routed": routed}
    model.logger.info(
        f"[R1A_INSTALL] rank={core.mapping.tp_rank} disabled={disabled} event_pairs={len(state.events)}"
    )
    return state


def _arm(model, multi, disabled):
    cls = model.MiniMaxM3Tp2ModelingV2
    original = cls.post_load_weights
    owner = getattr(original, "_r1a_owner", None)
    if owner is not None:
        if owner != (VERSION, id(model), disabled):
            raise RuntimeError("conflicting R1a installation")
        return

    @functools.wraps(original)
    def post_load(core):
        previous = core.__dict__.get("_r1a_state")
        if previous is not None:
            previous.check_thread()
            identity = tuple(core.w[k].data_ptr() for k in ("embed", "l0_in_norm", "final_norm"))
            if identity != previous.built_for:
                raise RuntimeError("weight replacement requires graph retirement before post-load")
        original(core)
        _install(model, core, multi, disabled)

    post_load._r1a_owner = (VERSION, id(model), disabled)
    cls.post_load_weights = post_load


def arm(model):
    if hashlib.md5(Path(model.__file__).read_bytes()).hexdigest() != MODEL_MD5:
        raise RuntimeError("R1A requires the pinned TP2 source")
    from tensorrt_llm._torch.modules import multi_stream_utils

    _arm(model, multi_stream_utils, model.m31_disabled("R1A"))
