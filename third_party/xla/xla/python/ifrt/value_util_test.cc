/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/value_util.h"

#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

TEST(ValueUtilTest, ToArraysCopy) {
  ValueRef array0 = tsl::MakeRef<MockArray>();
  ValueRef array1 = tsl::MakeRef<MockArray>();
  std::vector<ValueRef> values{array0, array1};
  std::vector<ArrayRef> arrays = ToArrays(absl::MakeConstSpan(values));
  ASSERT_EQ(arrays.size(), 2);
  EXPECT_EQ(arrays[0].get(), array0.get());
  EXPECT_EQ(arrays[1].get(), array1.get());
}

TEST(ValueUtilTest, ToArraysMove) {
  ValueRef array0 = tsl::MakeRef<MockArray>();
  ValueRef array1 = tsl::MakeRef<MockArray>();
  std::vector<ValueRef> values{array0, array1};
  ValueRef array0_ptr = array0;
  ValueRef array1_ptr = array1;
  std::vector<ArrayRef> arrays = ToArrays(absl::Span<ValueRef>(values));
  ASSERT_EQ(arrays.size(), 2);
  EXPECT_EQ(arrays[0].get(), array0_ptr.get());
  EXPECT_EQ(arrays[1].get(), array1_ptr.get());
  EXPECT_EQ(values[0].get(), nullptr);
  EXPECT_EQ(values[1].get(), nullptr);
}

TEST(ValueUtilTest, ToValuesCopy) {
  auto array1 = tsl::MakeRef<MockArray>();
  auto array2 = tsl::MakeRef<MockArray>();
  std::vector<ArrayRef> arrays = {array1, array2};
  std::vector<ValueRef> values = ToValues(arrays);
  EXPECT_EQ(values.size(), 2);
  EXPECT_EQ(values[0].get(), array1.get());
  EXPECT_EQ(values[1].get(), array2.get());
}

TEST(ValueUtilTest, ToValuesMove) {
  auto array1 = tsl::MakeRef<MockArray>();
  auto array2 = tsl::MakeRef<MockArray>();
  std::vector<ArrayRef> arrays = {array1, array2};
  auto* ptr1 = array1.get();
  auto* ptr2 = array2.get();
  std::vector<ValueRef> values = ToValues(absl::MakeSpan(arrays));
  EXPECT_EQ(values.size(), 2);
  EXPECT_EQ(values[0].get(), ptr1);
  EXPECT_EQ(values[1].get(), ptr2);
  EXPECT_EQ(arrays[0].get(), nullptr);
  EXPECT_EQ(arrays[1].get(), nullptr);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
