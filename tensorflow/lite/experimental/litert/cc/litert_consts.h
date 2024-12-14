// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_CONSTS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_CONSTS_H_

namespace litert {

static constexpr int kMaxExpectedTensorRank = 6;
static constexpr int kMaxExpectedTensorUses = 8;
static constexpr int kMaxExpectedOpInputs = 4;
static constexpr int kMaxExpectedOpOutputs = 8;
static constexpr int kMaxExpectedSubgraphInputs = 4;
static constexpr int kMaxExpectedSubgraphOutputs = 4;

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_CONSTS_H_
