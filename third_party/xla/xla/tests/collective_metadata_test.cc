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
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tests/collective_ops_e2e_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

class CollectiveMetadataTest : public CollectiveOpsE2ETestBase {
 protected:
  CollectiveMetadataTest()
      : CollectiveOpsE2ETestBase(/*memory_size=*/32 * kMB,
                                 /*collectives_memory_size=*/kMB) {}

  void SetUp() override {
    CollectiveOpsE2ETestBase::SetUp();
    if (!IsHopperAndHigher()) {
      GTEST_SKIP() << "Test requires Hopper or newer architecture since it's "
                      "using a multicast.";
    }
  }
};

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadata) {
  constexpr int kNumReplicas = 2;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> unoptimized_module,
                          ParseAndReturnVerifiedModule(R"(
  HloModule test, replica_count=2

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)
    copy_1 = f32[4]{0:S(1)} copy(param_1)

    const_0 = f32[1] constant({10})

    result_tuple = (f32[4], f32[4]{0:S(1)}, f32[1], u64[9]) custom-call(param_0, copy_1, const_0), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}
    ROOT get_tuple_element = u64[9] get-tuple-element(result_tuple), index=3
  })"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      hlo_runner_->CreateExecutable(std::move(unoptimized_module),
                                    /*run_hlo_passes=*/false));
  const std::array<Literal, 2> arguments = {
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f}),
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f})};
  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> result,
      ExecuteReplicated(
          /*executable_provider*/ [&](int64_t) { return executable.get(); },
          /*argument_count_provider*/ [&](int64_t) { return arguments.size(); },
          /*argument_provider*/
          [&](int64_t replica_id, int64_t arg_index) {
            return &arguments[arg_index];
          },
          kNumReplicas,
          /*run_hlo_passes=*/false, &device_assignment));

  ASSERT_EQ(result.size(), kNumReplicas);
  Literal first_result = std::move(result[0]);
  Literal second_result = std::move(result[1]);

  absl::Span<const uint64_t> first_result_data = first_result.data<uint64_t>();
  absl::Span<const uint64_t> second_result_data =
      second_result.data<uint64_t>();
  constexpr int kNumElements = 9;
  ASSERT_EQ(first_result_data.size(), kNumElements);
  ASSERT_EQ(second_result_data.size(), kNumElements);

  // Check the rank in the first position.
  EXPECT_EQ(first_result_data[0], 0);
  EXPECT_EQ(second_result_data[0], 1);

  // Check pointer to peers in the second position.
  EXPECT_NE(first_result_data[1], 0);
  EXPECT_NE(second_result_data[1], 0);

  // Check pointer to multimem metadata in the third position.
  EXPECT_NE(first_result_data[2], 0);
  EXPECT_NE(second_result_data[2], 0);

  // Check param_to_peers structure.
  for (int i = 3; i < kNumElements; ++i) {
    EXPECT_NE(first_result_data[i], 0);
    EXPECT_EQ(second_result_data[i], first_result_data[i]);
  }
}

TEST_F(CollectiveMetadataTest, ConstructCollectiveMetadataWithReplicaGroup) {
  constexpr int kNumReplicas = 4;
  if (hlo_runner_->device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires at least " << kNumReplicas << " devices ("
                 << hlo_runner_->device_count() << " available)";
  }
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> unoptimized_module,
                          ParseAndReturnVerifiedModule(R"(
  HloModule test, replica_count=4

  ENTRY test_computation {
    param_0 = f32[4] parameter(0)
    param_1 = f32[4] parameter(1)
    copy_1 = f32[4]{0:S(1)} copy(param_1)

    result_tuple = (f32[4], f32[4]{0:S(1)}, u64[7]) custom-call(param_0, copy_1), custom_call_target="CollectiveMetadata", output_to_operand_aliasing={{0}: (0, {}), {1}: (1, {})}, backend_config="{\"collective_metadata_backend_config\":{\"collective_devices\": { \"replica_groups\": [{\"replica_ids\": [0,1]}, {\"replica_ids\": [2,3]}]}}}"
    ROOT get_tuple_element = u64[7] get-tuple-element(result_tuple), index=2
  })"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OpaqueExecutable> executable,
      hlo_runner_->CreateExecutable(std::move(unoptimized_module),
                                    /*run_hlo_passes=*/false));
  const std::array<Literal, 2> arguments = {
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f}),
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f})};
  DeviceAssignment device_assignment = MakeDeviceAssignment(kNumReplicas);
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> result,
      ExecuteReplicated(
          /*executable_provider*/ [&](int64_t) { return executable.get(); },
          /*argument_count_provider*/ [&](int64_t) { return arguments.size(); },
          /*argument_provider*/
          [&](int64_t replica_id, int64_t arg_index) {
            return &arguments[arg_index];
          },
          kNumReplicas,
          /*run_hlo_passes=*/false, &device_assignment));

  ASSERT_EQ(result.size(), kNumReplicas);
  Literal replica_0_result_0 = std::move(result[0]);
  Literal replica_0_result_1 = std::move(result[1]);
  Literal replica_1_result_0 = std::move(result[2]);
  Literal replica_1_result_1 = std::move(result[3]);

  absl::Span<const uint64_t> replica_0_result_0_data =
      replica_0_result_0.data<uint64_t>();
  absl::Span<const uint64_t> replica_0_result_1_data =
      replica_0_result_1.data<uint64_t>();
  absl::Span<const uint64_t> replica_1_result_0_data =
      replica_1_result_0.data<uint64_t>();
  absl::Span<const uint64_t> replica_1_result_1_data =
      replica_1_result_1.data<uint64_t>();

  // Check the rank in the first position.
  constexpr int kNumElements = 7;
  ASSERT_EQ(replica_0_result_0_data.size(), kNumElements);
  ASSERT_EQ(replica_0_result_1_data.size(), kNumElements);
  ASSERT_EQ(replica_1_result_0_data.size(), kNumElements);
  ASSERT_EQ(replica_1_result_1_data.size(), kNumElements);

  EXPECT_EQ(replica_0_result_0_data[0], 0);
  EXPECT_EQ(replica_0_result_1_data[0], 1);
  EXPECT_EQ(replica_1_result_0_data[0], 0);
  EXPECT_EQ(replica_1_result_1_data[0], 1);

  // Check pointer to peers in the second position.
  EXPECT_NE(replica_0_result_0_data[1], 0);
  EXPECT_NE(replica_0_result_1_data[1], 0);
  EXPECT_NE(replica_1_result_0_data[1], 0);
  EXPECT_NE(replica_1_result_1_data[1], 0);

  // Check pointer to multimem metadata in the third position.
  EXPECT_NE(replica_0_result_0_data[2], 0);
  EXPECT_NE(replica_0_result_1_data[2], 0);
  EXPECT_NE(replica_1_result_0_data[2], 0);
  EXPECT_NE(replica_1_result_1_data[2], 0);

  // Check param_to_peers structure.
  for (int i = 3; i < kNumElements; ++i) {
    EXPECT_NE(replica_0_result_0_data[i], 0);
    EXPECT_EQ(replica_0_result_1_data[i], replica_0_result_0_data[i]);
    EXPECT_NE(replica_1_result_0_data[i], 0);
    EXPECT_EQ(replica_1_result_1_data[i], replica_1_result_0_data[i]);
  }
}

}  // namespace
}  // namespace xla
