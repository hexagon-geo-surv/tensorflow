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

#include <limits>

#include <gtest/gtest.h>
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

TEST(FP8UtilsTest, F8E4M3FNDistance) {
  // a & b are equal
  EXPECT_EQ(fp8_util::CalculateF8Distance(tsl::float8_e4m3fn(8.0),
                                          tsl::float8_e4m3fn(8.0)),
            0);

  // a & b have the same exponents
  EXPECT_EQ(fp8_util::CalculateF8Distance(tsl::float8_e4m3fn(8.0),
                                          tsl::float8_e4m3fn(13)),
            5);

  // a & b have different exponents
  EXPECT_EQ(fp8_util::CalculateF8Distance(tsl::float8_e4m3fn(8.0),
                                          tsl::float8_e4m3fn(6.0)),
            4);

  // 1 from 0 in the positive direction
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e4m3fn>::denorm_min(),
                tsl::float8_e4m3fn(0)),
            1);

  // 1 from 0 in the negative direction
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e4m3fn>::denorm_min() *
                    tsl::float8_e4m3fn(-1),
                tsl::float8_e4m3fn(0)),
            1);

  // a & b have different signs
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e4m3fn>::denorm_min(),
                std::numeric_limits<tsl::float8_e4m3fn>::denorm_min() *
                    tsl::float8_e4m3fn(-1)),
            2);

  // 1 non denorm from 0 in the positive direction
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e4m3fn>::min(),
                tsl::float8_e4m3fn(0)),
            8);

  // 1 non denorm from 0 in the negative direction
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e4m3fn>::min() *
                    tsl::float8_e4m3fn(-1),
                tsl::float8_e4m3fn(0)),
            8);

  // a & b have different signs
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e4m3fn>::min(),
                std::numeric_limits<tsl::float8_e4m3fn>::min() *
                    tsl::float8_e4m3fn(-1)),
            16);
}

TEST(FP8UtilsTest, F8E5M2Distance) {
  // a & b are equal
  EXPECT_EQ(fp8_util::CalculateF8Distance(tsl::float8_e5m2(8.0),
                                          tsl::float8_e5m2(8.0)),
            0);

  // a & b have the same exponents
  EXPECT_EQ(fp8_util::CalculateF8Distance(tsl::float8_e5m2(8.0),
                                          tsl::float8_e5m2(14)),
            3);

  // a & b have different exponents
  EXPECT_EQ(
      fp8_util::CalculateF8Distance(tsl::float8_e5m2(8.0), tsl::float8_e5m2(6)),
      2);

  // 1 from 0 in the positive direction
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e5m2>::denorm_min(),
                tsl::float8_e5m2(0)),
            1);

  // 1 from 0 in the negative direction
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e5m2>::denorm_min() *
                    tsl::float8_e5m2(-1),
                tsl::float8_e5m2(0)),
            1);

  // a & b have different signs
  EXPECT_EQ(fp8_util::CalculateF8Distance(
                std::numeric_limits<tsl::float8_e5m2>::denorm_min(),
                std::numeric_limits<tsl::float8_e5m2>::denorm_min() *
                    tsl::float8_e5m2(-1)),
            2);

  // 1 non denorm from 0 in the positive direction
  EXPECT_EQ(
      fp8_util::CalculateF8Distance(
          std::numeric_limits<tsl::float8_e5m2>::min(), tsl::float8_e5m2(0)),
      4);

  // 1 non denorm from 0 in the negative direction
  EXPECT_EQ(
      fp8_util::CalculateF8Distance(
          std::numeric_limits<tsl::float8_e5m2>::min() * tsl::float8_e5m2(-1),
          tsl::float8_e5m2(0)),
      4);

  // a & b have different signs
  EXPECT_EQ(
      fp8_util::CalculateF8Distance(
          std::numeric_limits<tsl::float8_e5m2>::min(),
          std::numeric_limits<tsl::float8_e5m2>::min() * tsl::float8_e5m2(-1)),
      8);
}

}  // namespace
}  // namespace xla
