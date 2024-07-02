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

#include "xla/fp8_util.h"

#include <cstdint>
#include <cstdlib>

#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace fp8_util {

int CalculateF8Distance(tsl::float8_e4m3fn a_f8, tsl::float8_e4m3fn b_f8) {
  uint8_t a = a_f8.rep();
  uint8_t b = b_f8.rep();

  auto zero = static_cast<tsl::float8_e4m3fn>(0);
  if (a_f8 == zero) {
    return std::abs(static_cast<int>(b & 0x7F));
  } else if (b_f8 == zero) {
    return std::abs(static_cast<int>(a & 0x7F));
  } else if ((a_f8 > zero && b_f8 > zero) || (a_f8 < zero && b_f8 < zero)) {
    return std::abs(static_cast<int>(a - b));
  } else {
    return std::abs(static_cast<int>(a & 0x7F)) +
           std::abs(static_cast<int>(b & 0x7F));
  }
  return 0;
}

int CalculateF8Distance(tsl::float8_e5m2 a_f8, tsl::float8_e5m2 b_f8) {
  uint8_t a = a_f8.rep();
  uint8_t b = b_f8.rep();

  auto zero = static_cast<tsl::float8_e5m2>(0);
  if (a_f8 == zero) {
    return std::abs(static_cast<int>(b & 0x7F));
  } else if (b_f8 == zero) {
    return std::abs(static_cast<int>(a & 0x7F));
  } else if ((a_f8 > zero && b_f8 > zero) || (a_f8 < zero && b_f8 < zero)) {
    return std::abs(static_cast<int>(a - b));
  } else {
    return std::abs(static_cast<int>(a & 0x7F)) +
           std::abs(static_cast<int>(b & 0x7F));
  }
  return 0;
}

}  // namespace fp8_util
}  // namespace xla
