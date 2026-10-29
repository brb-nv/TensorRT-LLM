# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT
"""SM100-family Q8KV4 paged decode indexer."""

from ....indexer.decode.q8kv4.interface import (
    BatchDecodeIndexerWithPagedKVCacheWrapper,
)

__all__ = ["BatchDecodeIndexerWithPagedKVCacheWrapper"]
