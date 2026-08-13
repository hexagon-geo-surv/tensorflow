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

#include "xla/tests/aot_compatibility_experimental/test_lib.h"

#include <stdlib.h>

#include <string>
#include <vector>

#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tsl/platform/test.h"
#include "xla/util/split_proto/human_readable_aot_executable.pb.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace aot_compatibility_experimental {
namespace {

using ::testing::ElementsAre;

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_With4Versions) {
  unsetenv("XLA_AOT_TEST_ALL_VERSIONS");
  ASSERT_OK_AND_ASSIGN(
      std::vector<AotTestParam> params,
      GetAotTestParamsForBackwardsCompatibility("test_dummy_test"));
  EXPECT_THAT(params,
              ElementsAre(AotTestParam{AOTTestMode::kBackwardsCompatibility, 1,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 3,
                                       "test_dummy_test"}));
}

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_AllVersions) {
  setenv("XLA_AOT_TEST_ALL_VERSIONS", "1", 1);
  ASSERT_OK_AND_ASSIGN(
      std::vector<AotTestParam> params,
      GetAotTestParamsForBackwardsCompatibility("test_dummy_test"));
  EXPECT_THAT(params,
              ElementsAre(AotTestParam{AOTTestMode::kBackwardsCompatibility, 1,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 2,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 3,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 4,
                                       "test_dummy_test"}));
  unsetenv("XLA_AOT_TEST_ALL_VERSIONS");
}

TEST(TestLibTest, GetAotTestParamsForGoldenFileVerification_With4Versions) {
  ASSERT_OK_AND_ASSIGN(
      std::vector<AotTestParam> params,
      GetAotTestParamsForGoldenFileVerification("test_dummy_test"));
  EXPECT_THAT(params, ElementsAre(AotTestParam{AOTTestMode::kGoldenVerification,
                                               4, "test_dummy_test"}));
}

TEST(TestLibTest, AOTInterceptionPjrtClientLoadSerializedArtifact_Succeeds) {
  std::string artifact_path = tsl::io::JoinPath(
      GetExecutablesDirectory("test_dummy_test"), "v1", "exec.pbtxt");
  AOTInterceptionPjrtClient client(
      nullptr, AOTTestMode::kBackwardsCompatibility, artifact_path);
  ASSERT_OK_AND_ASSIGN(std::string serialized, client.LoadSerializedArtifact());
  EXPECT_FALSE(serialized.empty());
}

TEST(TestLibTest, CompareGoldenGPUExecutable_NormalizesXlaDumpToAndReturnsOk) {
  HumanReadableAotExecutable golden;
  golden.mutable_gpu_executable()->set_binary("golden_binary");
  golden.mutable_gpu_executable()
      ->mutable_hlo_module_with_config()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_dump_to("/tmp/golden_dir");
  golden.mutable_gpu_executable()->set_asm_text("test_asm");

  HumanReadableAotExecutable fresh;
  fresh.mutable_gpu_executable()->set_binary(
      "fresh_binary_that_should_be_ignored");
  fresh.mutable_gpu_executable()
      ->mutable_hlo_module_with_config()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_dump_to("/tmp/fresh_dir");
  fresh.mutable_gpu_executable()->set_asm_text("completely_different_asm");

  EXPECT_OK(
      AOTInterceptionPjrtClient::CompareGoldenGPUExecutable(fresh, golden));
}

// TODO(b/528258781): Debug options are currently cleared wholesale before the
// structural comparison, so differing compiler flags are NOT detected. Once we
// decide which flags must be preserved, this test should assert that meaningful
// flag changes are detected again.
TEST(TestLibTest, CompareGoldenGPUExecutable_IgnoresDebugOptionsForNow) {
  HumanReadableAotExecutable golden;
  golden.mutable_gpu_executable()->set_binary("same_binary");
  golden.mutable_gpu_executable()
      ->mutable_hlo_module_with_config()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_gpu_enable_fast_min_max(true);

  HumanReadableAotExecutable fresh;
  fresh.mutable_gpu_executable()->set_binary("same_binary");
  fresh.mutable_gpu_executable()
      ->mutable_hlo_module_with_config()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_gpu_enable_fast_min_max(false);

  EXPECT_OK(
      AOTInterceptionPjrtClient::CompareGoldenGPUExecutable(fresh, golden));
}

// The compile-options copy of debug_options must also be cleared before the
// structural comparison; a flag difference there must not cause a spurious
// mismatch. See b/528258781.
TEST(TestLibTest,
     CompareGoldenGPUExecutable_IgnoresCompileOptionsDebugOptionsForNow) {
  HumanReadableAotExecutable golden;
  golden.mutable_gpu_executable()->set_binary("same_binary");
  golden.mutable_executable_and_options()
      ->mutable_compile_options()
      ->mutable_executable_build_options()
      ->mutable_debug_options()
      ->set_xla_gpu_enable_fast_min_max(true);

  HumanReadableAotExecutable fresh;
  fresh.mutable_gpu_executable()->set_binary("same_binary");
  fresh.mutable_executable_and_options()
      ->mutable_compile_options()
      ->mutable_executable_build_options()
      ->mutable_debug_options()
      ->set_xla_gpu_enable_fast_min_max(false);

  EXPECT_OK(
      AOTInterceptionPjrtClient::CompareGoldenGPUExecutable(fresh, golden));
}

// A different host-specific debug option (CUDA install path) must be normalized
// away and must not cause a spurious mismatch.
TEST(TestLibTest,
     CompareGoldenGPUExecutable_NormalizesHostPathFieldsAndReturnsOk) {
  HumanReadableAotExecutable golden;
  golden.mutable_gpu_executable()->set_binary("same_binary");
  golden.mutable_gpu_executable()
      ->mutable_hlo_module_with_config()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_gpu_cuda_data_dir("/host_a/cuda");

  HumanReadableAotExecutable fresh;
  fresh.mutable_gpu_executable()->set_binary("same_binary");
  fresh.mutable_gpu_executable()
      ->mutable_hlo_module_with_config()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_gpu_cuda_data_dir("/host_b/cuda");

  EXPECT_OK(
      AOTInterceptionPjrtClient::CompareGoldenGPUExecutable(fresh, golden));
}

TEST(TestLibTest, CompareGoldenGPUExecutable_FailsOnGenuineDifferences) {
  HumanReadableAotExecutable golden;
  golden.mutable_gpu_executable()->set_binary("same_binary");
  golden.mutable_gpu_executable()->set_module_name("test_module");

  HumanReadableAotExecutable fresh;
  fresh.mutable_gpu_executable()->set_binary("same_binary");
  fresh.mutable_gpu_executable()->set_module_name("different_module");

  EXPECT_FALSE(
      AOTInterceptionPjrtClient::CompareGoldenGPUExecutable(fresh, golden)
          .ok());
}

}  // namespace
}  // namespace aot_compatibility_experimental
}  // namespace xla
