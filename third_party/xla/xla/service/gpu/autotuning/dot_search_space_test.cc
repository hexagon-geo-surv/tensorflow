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

#include "xla/service/gpu/autotuning/dot_search_space.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::Eq;
using ::testing::Field;
using ::testing::Ge;
using ::testing::IsEmpty;
using ::testing::Le;

#define DEFINE_FIELD_MATCHER(matcher_name, field_name)                 \
  auto matcher_name(auto matcher) {                                    \
    return Field(#field_name, &TritonGemmConfig::field_name, matcher); \
  }
// We technically don't need the semicolons below, but it throws off the VSCode
// formatter if we have what looks like a statement without a semicolon, so
// adding it makes editing the file a lot easier.
DEFINE_FIELD_MATCHER(BlockMIs, block_m);
DEFINE_FIELD_MATCHER(BlockNIs, block_n);
DEFINE_FIELD_MATCHER(BlockKIs, block_k);
DEFINE_FIELD_MATCHER(SplitKIs, split_k);
DEFINE_FIELD_MATCHER(NumStagesIs, num_stages);
DEFINE_FIELD_MATCHER(NumWarpsIs, num_warps);
DEFINE_FIELD_MATCHER(NumCtasIs, num_ctas);
#undef DEFINE_FIELD_MATCHER

auto IsValidConfig() {
  return AllOf(BlockMIs(Ge(1)), BlockNIs(Ge(1)), BlockKIs(Ge(1)),
               SplitKIs(Ge(1)), NumStagesIs(Ge(1)), NumWarpsIs(Ge(1)),
               NumCtasIs(Ge(1)));
};

class DotSearchSpaceTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_description_{
      se::DeviceDescription(se::GpuDeviceInfoProto::default_instance())};

  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> GetDefaultDotModule(
      int lhs_parallel_dim = 1024, int rhs_parallel_dim = 1024,
      int contracting_dim = 1024) {
    constexpr const char* kModuleTextFormat = R"(
ENTRY e {
  p0 = f16[%d,%d] parameter(0)
  p1 = f16[%d,%d] parameter(1)
  ROOT r = f16[%d,%d] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";
    return ParseAndReturnVerifiedModule(absl::StrFormat(
        kModuleTextFormat, lhs_parallel_dim, contracting_dim, contracting_dim,
        rhs_parallel_dim, lhs_parallel_dim, rhs_parallel_dim));
  }

  HloDotInstruction* GetDot(VerifiedHloModule* module) {
    return Cast<HloDotInstruction>(
        module->entry_computation()->root_instruction());
  }
};

TEST_F(DotSearchSpaceTest, ReturnsValidConfigList) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space(device_description_,
                                          GetDot(module.get()));

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(IsValidConfig())));
}

TEST_F(DotSearchSpaceTest, HonorsForcedContractingSplit) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetDefaultDotModule());
  TritonDotFusionSearchSpace search_space(device_description_,
                                          GetDot(module.get()));

  EXPECT_THAT(
      search_space.GenerateConfigs(/*force_contracting_split=*/2),
      AllOf(Not(IsEmpty()), Each(IsValidConfig()), Each(SplitKIs(Eq(2)))));
}

TEST_F(DotSearchSpaceTest, ConsidersContractingSplitForSmallOutputSize) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/16,
                                              /*rhs_parallel_dim=*/16,
                                              /*contracting_dim=*/1024));
  TritonDotFusionSearchSpace search_space(device_description_,
                                          GetDot(module.get()));

  EXPECT_THAT(search_space.GenerateConfigs(), Contains(SplitKIs(Ge(2))));
}

TEST_F(DotSearchSpaceTest, LimitsContractingSplitForSmallerContractingSize) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetDefaultDotModule(/*lhs_parallel_dim=*/16,
                                              /*rhs_parallel_dim=*/16,
                                              /*contracting_dim=*/32));
  TritonDotFusionSearchSpace search_space(device_description_,
                                          GetDot(module.get()));

  EXPECT_THAT(search_space.GenerateConfigs(),
              AllOf(Not(IsEmpty()), Each(SplitKIs(Le(2)))));
}

}  // namespace
}  // namespace xla::gpu
