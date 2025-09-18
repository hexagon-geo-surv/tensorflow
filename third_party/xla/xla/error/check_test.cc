/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/error/check.h"

#include <gtest/gtest.h>
#include "xla/error/debug_me_context_util.h"
#include "xla/tsl/platform/debug_me_context.h"

namespace xla::error {
namespace {

TEST(CheckTest, TrueCondition_DoesNotCrash) { XLA_CHECK(true); }

TEST(CheckTest, TrueConditionWithExpression_EvaluatesOnlyOnce) {
  int x = 0;
  XLA_CHECK(++x == 1);
  EXPECT_EQ(x, 1);
}

TEST(CheckTest, TrueConditionWithMessageExpression_DoesNotEvaluate) {
  int x = 0;
  XLA_CHECK(true) << ++x;
  EXPECT_EQ(x, 0);
}

TEST(CheckTest, FalseCondition_Crashes) {
  EXPECT_DEATH(XLA_CHECK(false), "check_test.cc:\\d+] Check failed: false");
}

TEST(CheckTest, FalseConditionWithMessage_Crashes) {
  EXPECT_DEATH(XLA_CHECK(false) << "custom error message",
               "check_test.cc:\\d+] Check failed: false custom error message");
}

TEST(CheckTest, FalseConditionWithExpression_Crashes) {
  int x = 0;
  EXPECT_DEATH(XLA_CHECK(x == 1), "check_test.cc:\\d+] Check failed: x == 1");
}

TEST(CheckTest, FalseCondition_WithMessageAndDebugContext_Crashes) {
  tsl::DebugMeContext<DebugMeContextKey> context(DebugMeContextKey::kHloPass,
                                                 "MyTestPass");
  EXPECT_DEATH(XLA_CHECK(false) << "custom error message",
               "check_test.cc:\\d+] Check failed: false custom error "
               "message\nDebugMeContext:\nHLO Passes: MyTestPass");
}

}  // namespace
}  // namespace xla::error
