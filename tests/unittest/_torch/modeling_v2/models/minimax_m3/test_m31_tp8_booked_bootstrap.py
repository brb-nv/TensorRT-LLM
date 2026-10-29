# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""CPU dispatch checks for the booked TP8 factory, without CUDA imports."""

import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
PACKAGE = (
    ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m31_local_moe_tp8"
)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.name = "_tp8_booked_cpu"
        package = types.ModuleType(self.name)
        package.__path__ = [str(PACKAGE)]
        self.local = types.ModuleType(self.name + ".m31_local_moe_op")
        self.local.make_state = Mock(return_value=object())
        self.modules = patch.dict(
            sys.modules, {self.name: package, self.local.__name__: self.local}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.bootstrap = importlib.import_module(self.name + ".bootstrap")

    def model(self):
        return NS(M3MoeState=Mock(return_value=object()), _ENGINE_STATES={})

    def test_only_booked_shape_uses_local_factory_on_each_rank(self):
        for rank in range(8):
            model = self.model()
            stock = model.M3MoeState
            self.bootstrap.install_tp8(model, NS(tp_size=8, tp_rank=rank))
            self.assertIs(model.M3MoeState(0, 8, shared=True), self.local.make_state.return_value)
            self.local.make_state.assert_called_with(0, 8, num_ctas=None, shared=True)
            for tokens, shared in ((8, False), (16, True), (1, True), (176, True)):
                self.assertIs(model.M3MoeState(0, tokens, shared=shared), stock.return_value)
            self.assertEqual(stock.call_count, 4)

    def test_idempotent_install_and_rank_change(self):
        model = self.model()
        mapping = NS(tp_size=8, tp_rank=0)
        self.bootstrap.install_tp8(model, mapping)
        factory = model.M3MoeState
        self.bootstrap.install_tp8(model, mapping)
        self.assertIs(model.M3MoeState, factory)
        with self.assertRaisesRegex(RuntimeError, "rank changed"):
            self.bootstrap.install_tp8(model, NS(tp_size=8, tp_rank=1))

    def test_no_tp2_install(self):
        model = self.model()
        stock = model.M3MoeState
        with self.assertRaisesRegex(ValueError, "requires TP8 rank"):
            self.bootstrap.install_tp8(model, NS(tp_size=2, tp_rank=0))
        self.assertIs(model.M3MoeState, stock)
        self.local.make_state.assert_not_called()

    def test_preserves_existing_pre_engine_install_requirement(self):
        model = self.model()
        model._ENGINE_STATES["already-built"] = object()
        stock = model.M3MoeState
        with self.assertRaisesRegex(RuntimeError, "follows engine state build"):
            self.bootstrap.install_tp8(model, NS(tp_size=8, tp_rank=0))
        self.assertIs(model.M3MoeState, stock)
        self.local.make_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
