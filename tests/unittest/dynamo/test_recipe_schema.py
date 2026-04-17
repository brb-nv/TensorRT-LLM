# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""L0 tests for validating Dynamo TRTLLM recipe YAML configs against TorchLlmArgs.

The Dynamo project (https://github.com/ai-dynamo/dynamo) ships TRTLLM serving
recipes whose ConfigMaps are consumed by ``trtllm-serve --config``. This test
validates the in-tree copies of those recipes against the current
``TorchLlmArgs`` Pydantic schema so that TRTLLM-side changes (field renames,
type changes, removed options, CLI wiring) fail fast in pre-merge CI.

These tests are CPU-only: CUDA is mocked via ``mock_cuda_for_schema_validation``
and ``PyTorchLLM`` / ``OpenAIServer`` are mocked out in the serve-CLI test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

# The shared YAML-validation harness lives under tests/unittest/llmapi. That
# package uses an intra-package relative import in its own tests, so add the
# llmapi directory to sys.path and import the harness module directly.
_HARNESS_DIR = Path(__file__).resolve().parents[1] / "llmapi"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from yaml_validation_harness import (  # noqa: E402
    collect_yaml_files,
    load_yaml_dict,
    mock_cuda_for_schema_validation,
    validate_torch_llm_args_config,
)

from tensorrt_llm.commands.serve import main as serve_main  # noqa: E402
from tensorrt_llm.llmapi import llm_args as llm_args_module  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DYNAMO_CONFIGS_DIR = REPO_ROOT / "tests" / "integration" / "defs" / "dynamo" / "configs"

# All recipe YAMLs: agg (config.yaml) + disagg (config-prefill.yaml,
# config-decode.yaml) across every model directory.
DYNAMO_RECIPE_CONFIGS = collect_yaml_files(DYNAMO_CONFIGS_DIR, "**/*.yaml")


def _config_id(config_path: Path) -> str:
    return str(config_path.relative_to(DYNAMO_CONFIGS_DIR))


@pytest.fixture(autouse=True)
def mock_gpu_environment():
    """Mock GPU functions for CPU-only schema test execution."""
    with mock_cuda_for_schema_validation():
        yield


@pytest.mark.part0
def test_dynamo_recipe_configs_discovered():
    """At least one Dynamo recipe YAML must be present under the configs tree."""
    assert DYNAMO_RECIPE_CONFIGS, (
        f"No Dynamo recipe YAML files found under {DYNAMO_CONFIGS_DIR}. "
        "Expected config.yaml / config-prefill.yaml / config-decode.yaml files."
    )


@pytest.mark.part0
@pytest.mark.parametrize("config_path", DYNAMO_RECIPE_CONFIGS, ids=_config_id)
def test_dynamo_recipe_validates_against_llm_args(config_path: Path):
    """Each recipe YAML must parse and validate against TorchLlmArgs.

    Exercises the exact merge path used by ``trtllm-serve`` at
    ``tensorrt_llm/commands/serve.py``: ``update_llm_args_with_extra_dict``
    followed by ``TorchLlmArgs(**merged)``.
    """
    config_dict = load_yaml_dict(config_path)
    validate_torch_llm_args_config(config_dict)


def _serve_cli_args(config_path: Path, port: int = 17999):
    """CLI argv for serve, like: trtllm-serve <model> --config <config.yaml> --port <port>."""
    return [
        "dummy/model",
        "--config",
        str(config_path),
        "--port",
        str(port),
    ]


async def _noop_serve(_host, _port, _sockets=None):
    pass


class _MockOpenAIServer:
    def __init__(self, generator, model, **kwargs):
        self.generator = generator
        self.model = model

    def __call__(self, host, port, sockets=None):
        return _noop_serve(host, port, sockets)


@pytest.mark.part0
@pytest.mark.parametrize("config_path", DYNAMO_RECIPE_CONFIGS, ids=_config_id)
def test_dynamo_recipe_serve_cli(config_path: Path):
    """Invoke ``trtllm-serve`` via its CLI with each recipe; server/LLM mocked.

    This covers the Click-level argument wiring so that a recipe YAML missing,
    renamed, or type-mismatched against the CLI surface also fails in pre-merge.
    """
    mock_llm = mock.Mock()
    mock_pytorch_llm = mock.Mock(return_value=mock_llm)

    with (
        mock.patch("tensorrt_llm.commands.serve.get_is_diffusion_model", return_value=False),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.commands.serve.PyTorchLLM", mock_pytorch_llm),
        mock.patch("tensorrt_llm.commands.serve.OpenAIServer", _MockOpenAIServer),
    ):
        serve_main(args=_serve_cli_args(config_path), standalone_mode=False)

    mock_pytorch_llm.assert_called_once()
    call_kwargs = mock_pytorch_llm.call_args[1]
    llm_args_module.TorchLlmArgs(**call_kwargs)
