/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Host-side arch sets of the MiniMax-M3 optimizations first enabled on Rubin (SM107) and widened to Blackwell
// (SM100/SM103). The C++ mirror of tensorrt_llm/_torch/blackwell_rubin_opts.py:
//
// - SM107 is always in the set, so Rubin behaves exactly as before.
// - SM100 and SM103 come from the gate's own `*_ARCHS` env var (a comma-separated SM list such as "100,103,107";
//   "sm_103a" / "103f" spellings are accepted), or from the gate's default when the var is unset. Only 100 and 103
//   are taken from it.
// - TRTLLM_M3_BLACKWELL_RUBIN_OPTS=0 (the umbrella kill switch) drops SM100/SM103 from every widened gate.
//
// Callers keep the answer in a function-local static, so each gate reads its env vars once per process (one GPU per
// rank).

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cctype>
#include <cstdlib>
#include <initializer_list>
#include <string>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

namespace blackwell_rubin_opts
{

constexpr int kRubinSm = 107;

// Whether TRTLLM_M3_BLACKWELL_RUBIN_OPTS leaves the Blackwell widenings on (unset or anything but "0").
inline bool umbrellaOn()
{
    char const* env = std::getenv("TRTLLM_M3_BLACKWELL_RUBIN_OPTS");
    return env == nullptr || std::string(env) != "0";
}

// Whether the comma-separated SM list `raw` names `sm` (entries like "103", "sm_103a", "103f"; unparsable entries
// are ignored).
inline bool archListHas(char const* raw, int sm)
{
    std::string const list(raw);
    size_t pos = 0;
    while (pos <= list.size())
    {
        size_t end = list.find_first_of(",;", pos);
        if (end == std::string::npos)
        {
            end = list.size();
        }
        std::string item = list.substr(pos, end - pos);
        // Trim, lower-case, drop an "sm_" prefix and trailing a/f suffixes.
        size_t const first = item.find_first_not_of(" \t");
        size_t const last = item.find_last_not_of(" \t");
        item = first == std::string::npos ? std::string() : item.substr(first, last - first + 1);
        for (auto& c : item)
        {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        if (item.rfind("sm_", 0) == 0)
        {
            item = item.substr(3);
        }
        while (!item.empty() && (item.back() == 'a' || item.back() == 'f'))
        {
            item.pop_back();
        }
        if (!item.empty() && item.find_first_not_of("0123456789") == std::string::npos && std::stoi(item) == sm)
        {
            return true;
        }
        pos = end + 1;
    }
    return false;
}

// Whether a widened gate runs on `sm`: SM107 always; SM100/SM103 where `archsEnv` (else `defaultBlackwell`) lists
// them, unless the umbrella switch is off; never on other SMs.
inline bool gateOn(char const* archsEnv, std::initializer_list<int> defaultBlackwell, int sm)
{
    if (sm == kRubinSm)
    {
        return true;
    }
    if ((sm != 100 && sm != 103) || !umbrellaOn())
    {
        return false;
    }
    char const* raw = std::getenv(archsEnv);
    if (raw != nullptr)
    {
        return archListHas(raw, sm);
    }
    for (int const listed : defaultBlackwell)
    {
        if (listed == sm)
        {
            return true;
        }
    }
    return false;
}

// gateOn for the current device.
inline bool gateOn(char const* archsEnv, std::initializer_list<int> defaultBlackwell = {100, 103})
{
    return gateOn(archsEnv, defaultBlackwell, getSMVersion());
}

} // namespace blackwell_rubin_opts

} // namespace common

TRTLLM_NAMESPACE_END
