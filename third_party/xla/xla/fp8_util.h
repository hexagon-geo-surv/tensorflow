/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_FP8_UTIL_H_
#define XLA_FP8_UTIL_H_

#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace fp8_util {

int CalculateF8Distance(tsl::float8_e5m2 a_f8, tsl::float8_e5m2 b_f8);
int CalculateF8Distance(tsl::float8_e4m3fn a_f8, tsl::float8_e4m3fn b_f8);

}  // namespace fp8_util
}  // namespace xla

#endif  // XLA_FP8_UTIL_H_
