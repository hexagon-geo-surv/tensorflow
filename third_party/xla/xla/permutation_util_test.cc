/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/permutation_util.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "xla/hlo/testlib/test.h"

namespace xla {
namespace {

TEST(PermutationUtilTest, IsPermutation_TrueCases) {
  EXPECT_TRUE(IsPermutation({}));
  EXPECT_TRUE(IsPermutation({0}));
  EXPECT_TRUE(IsPermutation({0, 1}));
  EXPECT_TRUE(IsPermutation({1, 0}));
  EXPECT_TRUE(IsPermutation({3, 1, 0, 2}));
}

TEST(PermutationUtilTest, IsPermutation_FalseCases) {
  EXPECT_FALSE(IsPermutation({-3}));
  EXPECT_FALSE(IsPermutation({1, 1}));
  EXPECT_FALSE(IsPermutation({3, 0, 2}));
}

TEST(PermutationUtilTest, IsIdentityPermutation_TrueCases) {
  EXPECT_TRUE(IsIdentityPermutation({}));
  EXPECT_TRUE(IsIdentityPermutation({0}));
  EXPECT_TRUE(IsIdentityPermutation({0, 1}));
  EXPECT_TRUE(IsIdentityPermutation({0, 1, 2}));
  EXPECT_TRUE(IsIdentityPermutation({0, 1, 2, 3}));
}

TEST(PermutationUtilTest, IsIdentityPermutation_FalseCases) {
  std::vector<int> v{0, 1, 2, 3};
  std::next_permutation(v.begin(), v.end());

  do {
    EXPECT_FALSE(IsIdentityPermutation(v));
  } while (std::next_permutation(v.begin(), v.end()));
}

TEST(PermutationUtilTest, PermuteInverse) {
  EXPECT_EQ(PermuteInverse<std::vector<std::string>>({}, {}),
            (std::vector<std::string>{}));
  EXPECT_EQ(
      PermuteInverse<std::vector<std::string>>({"a", "b", "c"}, {0, 1, 2}),
      (std::vector<std::string>{"a", "b", "c"}));
  EXPECT_EQ(
      PermuteInverse<std::vector<std::string>>({"a", "b", "c"}, {2, 1, 0}),
      (std::vector<std::string>{"c", "b", "a"}));
  EXPECT_EQ(
      PermuteInverse<std::vector<std::string>>({"a", "b", "c"}, {2, 0, 1}),
      (std::vector<std::string>{"b", "c", "a"}));
}

TEST(PermutationUtilTest, InversePermutation) {
  EXPECT_EQ(InversePermutation({}), (std::vector<int64_t>{}));
}

TEST(PermutationUtilTest, ComposePermutations) {
  EXPECT_EQ(ComposePermutations({0, 1, 2}, {1, 2, 0}),
            (std::vector<int64_t>{1, 2, 0}));
  EXPECT_EQ(ComposePermutations({1, 2, 0}, {0, 1, 2}),
            (std::vector<int64_t>{1, 2, 0}));
  EXPECT_EQ(ComposePermutations({1, 3, 2, 0}, {2, 1, 3, 0}),
            (std::vector<int64_t>{2, 3, 0, 1}));
}

TEST(PermutationUtilTest, ComposeAndInversePermutations) {
  std::vector<int64_t> id{0, 1, 2, 3};
  std::vector<int64_t> p = id;

  do {
    EXPECT_EQ(ComposePermutations(InversePermutation(p), p), id);
  } while (std::next_permutation(p.begin(), p.end()));
}

}  // namespace
}  // namespace xla
