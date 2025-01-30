/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/shape_and_size_utils.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {
namespace {

TEST(SizeAndShapeUtilTest, TestConvertsSize) {
  EXPECT_EQ(ConvertToTfliteSize(1), 1);
  EXPECT_EQ(ConvertToTfliteSize(-1), -1);
  EXPECT_EQ(ConvertToTfliteSize(mlir::ShapedType::kDynamic), -1);
}

TEST(SizeAndShapeUtilTest, TestGetQuantDimensionAfterReshape) {
  // Test that the input quantization dimension is part of the folded/split
  // dimensions.
  EXPECT_EQ(GetQuantDimensionAfterReshape({1, 2, 3, 4, 5}, {24, 5}, 4), 1);
  EXPECT_EQ(GetQuantDimensionAfterReshape({1, 2, 3, 4, 5}, {1, 24, 5}, 4), 2);
  EXPECT_EQ(GetQuantDimensionAfterReshape({1, 2, 3, 4, 5}, {1, 2, 12, 5}, 4),
            3);
  EXPECT_EQ(GetQuantDimensionAfterReshape({1, 2, 3, 4, 5}, {2, 12, 5}, 4), 2);
  EXPECT_EQ(GetQuantDimensionAfterReshape({1, 2, 3, 5, 5}, {30, 5}, 4), 1);
  EXPECT_EQ(GetQuantDimensionAfterReshape({1, 2, 3, 4, 5}, {1, 6, 2, 2, 5}, 4),
            4);
  EXPECT_EQ(GetQuantDimensionAfterReshape({1, 2, 3, 4, 5}, {6, 2, 2, 5}, 4), 3);
  EXPECT_EQ(
      GetQuantDimensionAfterReshape({1, 2, 3, 4, 5}, {1, 2, 3, 2, 2, 5}, 4), 5);
  EXPECT_EQ(GetQuantDimensionAfterReshape({5, 1, 1, 8}, {1, 5, 1, 8}, 3), 3);
  EXPECT_EQ(GetQuantDimensionAfterReshape({5, 1, 1, 8}, {1, 1, 5, 8}, 3), 3);
  EXPECT_EQ(GetQuantDimensionAfterReshape({5, 1, 1, 8}, {1, 1, 5, 8}, 0), 2);
}
}  // namespace
}  // namespace TFL
}  // namespace mlir
