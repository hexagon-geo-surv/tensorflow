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

#include "xla/service/axl_transform.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_axl_transform_extension.h"
#include "xla/pjrt/c/pjrt_c_api_axl_transform_internal.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cse.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

class AxlTransformTest : public ::testing::Test {
 protected:
  void SetUp() override { ClearHloAxlTransforms(); }
  void TearDown() override { ClearHloAxlTransforms(); }
};

TEST_F(AxlTransformTest, Basic) {
  HloAxlTransform axl;
  auto status_or_bool = axl.Transform(nullptr);
  TF_ASSERT_OK(status_or_bool.status());
  EXPECT_FALSE(status_or_bool.value());
}

TEST_F(AxlTransformTest, Registration) {
  auto axl = std::make_shared<HloAxlTransform>();
  RegisterHloAxlTransform(HloAxlTransform::PipelineStage::kPreScheduler,
                          "test_transform", axl);

  const auto& transforms =
      GetHloAxlTransforms(HloAxlTransform::PipelineStage::kPreScheduler);
  ASSERT_EQ(transforms.size(), 1);
  EXPECT_EQ(transforms[0].first, "test_transform");
  EXPECT_NE(transforms[0].second, nullptr);
}

TEST_F(AxlTransformTest, ApplyTransforms) {
  absl::string_view hlo_text = R"(
    HloModule test_module
    ENTRY test_computation {
      p0 = f32[] parameter(0)
      ROOT add = f32[] add(p0, p0)
    }
  )";

  auto module_or_status = xla::ParseAndReturnUnverifiedModule(hlo_text);
  ASSERT_OK(module_or_status.status());
  auto module = std::move(module_or_status.value());

  class TrivialTransform : public HloAxlTransform {
   public:
    absl::StatusOr<bool> Transform(xla::HloModule* module) override {
      return true;
    }
  };

  auto transform = std::make_shared<TrivialTransform>();
  RegisterHloAxlTransform(HloAxlTransform::PipelineStage::kPreScheduler,
                          "trivial_transform", transform);

  auto status_or_bool = ApplyAxlTransformsToModule(
      HloAxlTransform::PipelineStage::kPreScheduler, module.get());
  TF_ASSERT_OK(status_or_bool.status());
  EXPECT_TRUE(status_or_bool.value());
}

TEST_F(AxlTransformTest, PassPipeline) {
  class TrivialTransform : public HloAxlTransform {
   public:
    absl::StatusOr<bool> Transform(xla::HloModule* module) override {
      return true;
    }
  };

  auto pre_transform = std::make_shared<TrivialTransform>();
  RegisterHloAxlTransform(HloAxlTransform::PipelineStage::kPreScheduler,
                          "pre_trivial", pre_transform);

  auto post_transform = std::make_shared<TrivialTransform>();
  RegisterHloAxlTransform(HloAxlTransform::PipelineStage::kPostScheduler,
                          "post_trivial", post_transform);

  HloPassPipeline pipeline("test_pipeline");

  AlgebraicSimplifierOptions options;
  pipeline.AddPass<AlgebraicSimplifier>(options);
  pipeline.AddPass<ApplyAxlTransforms>(
      HloAxlTransform::PipelineStage::kPreScheduler);
  pipeline.AddPass<HloTrivialScheduler>();
  pipeline.AddPass<ApplyAxlTransforms>(
      HloAxlTransform::PipelineStage::kPostScheduler);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);

  const char* hlo_text = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT neg = f32[] negate(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_text));

  auto status_or_bool = pipeline.Run(module.get());
  TF_ASSERT_OK(status_or_bool.status());
  EXPECT_TRUE(status_or_bool.value());
}

TEST_F(AxlTransformTest, PjrtCApiExtension) {
  // 1. Define the C callback.
  auto c_callback = [](PJRT_AxlTransform_Callbacks* callbacks,
                       PJRT_AxlTransform_Args* args) {
    EXPECT_NE(args->hlo_module.data, nullptr);
    EXPECT_GT(args->hlo_module.size, 0);

    xla::HloModuleProto proto;
    EXPECT_TRUE(
        proto.ParseFromArray(args->hlo_module.data, args->hlo_module.size));

    DebugOptions debug_options;
    auto config_or_status =
        HloModule::CreateModuleConfigFromProto(proto, debug_options);
    TF_ASSERT_OK(config_or_status.status());
    auto module_or_status =
        HloModule::CreateFromProto(proto, config_or_status.value());
    EXPECT_TRUE(module_or_status.status().ok());
    auto module = std::move(module_or_status.value());

    // Modify the module by adding a Negate instruction.
    auto* root = module->entry_computation()->root_instruction();
    auto* negate = module->entry_computation()->AddInstruction(
        HloInstruction::CreateUnary(root->shape(), HloOpcode::kNegate, root));
    module->entry_computation()->set_root_instruction(negate);

    xla::HloModuleProto modified_proto = module->ToProto();
    static thread_local std::string persistent_proto;
    persistent_proto.clear();
    EXPECT_TRUE(
        tsl::SerializeToStringDeterministic(modified_proto, &persistent_proto));

    args->changed = true;
    args->transformed_hlo_module.data = persistent_proto.data();
    args->transformed_hlo_module.size = persistent_proto.size();

    args->header.has_error = false;
  };

  // 2. Create the PJRT_AxlTransform_Callbacks struct.
  PJRT_AxlTransform_Callbacks callbacks;
  callbacks.version = PJRT_API_AXL_TRANSFORM_EXTENSION_VERSION;
  callbacks.dtor = nullptr;
  callbacks.transform_hlo_module = c_callback;

  // 3. Create the AXL transform extension.
  PJRT_Axl_Transform_Extension extension = pjrt::CreateAxlTransformExtension();

  // 4. Register the transform using the extension.
  PJRT_Register_Axl_Transform_Args args;
  args.struct_size = PJRT_Register_Axl_Transform_Args_STRUCT_SIZE;
  args.name = "pjrt_c_api_transform";
  args.name_size = sizeof("pjrt_c_api_transform") - 1;
  args.stage = PJRT_AxlTransform_PipelineStage_kPreScheduler;
  args.callbacks = &callbacks;

  PJRT_Error* error = extension.register_axl_transform(&args);
  EXPECT_EQ(error, nullptr);

  absl::string_view hlo_text = R"(
    HloModule test_module
    ENTRY test_computation {
      p0 = f32[] parameter(0)
      ROOT add = f32[] add(p0, p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_text));

  LOG(INFO) << "HloModule before AXL transforms:\n" << module->ToString();

  auto status_or_bool = ApplyAxlTransformsToModule(
      HloAxlTransform::PipelineStage::kPreScheduler, module.get());

  TF_ASSERT_OK(status_or_bool.status());
  EXPECT_TRUE(status_or_bool.value());

  LOG(INFO) << "HloModule after AXL transforms:\n" << module->ToString();
}

}  // namespace

}  // namespace xla
