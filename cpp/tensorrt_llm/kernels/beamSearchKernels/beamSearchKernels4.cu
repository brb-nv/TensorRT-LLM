/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "beamSearchKernelsTemplate.h"

namespace tensorrt_llm
{
namespace kernels
{
INSTANTIATE_BEAM_SEARCH(float, 4, false);
INSTANTIATE_BEAM_SEARCH(float, 4, true);
INSTANTIATE_BEAM_SEARCH(half, 4, false);
INSTANTIATE_BEAM_SEARCH(half, 4, true);
} // namespace kernels
} // namespace tensorrt_llm
