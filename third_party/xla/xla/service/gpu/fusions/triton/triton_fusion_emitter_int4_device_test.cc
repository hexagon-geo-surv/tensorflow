/* Copyright 2023 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class TritonTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Always rewrite Gemms with Triton regardless of size.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    return debug_options;
  }

  stream_executor::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  const stream_executor::GpuComputeCapability& GpuComputeComp() {
    return device_desc().gpu_compute_capability();
  }
  stream_executor::GpuComputeCapability CudaAmpereOrRocm() {
    if (std::holds_alternative<stream_executor::RocmComputeCapability>(
            GpuComputeComp())) {
      return stream_executor::GpuComputeCapability{
          device_desc().rocm_compute_capability()};
    } else {
      return stream_executor::GpuComputeCapability{
          stream_executor::CudaComputeCapability{
              stream_executor::CudaComputeCapability::AMPERE, 0}};
    }
  }

 protected:
  const stream_executor::DeviceDescription& device_desc() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
};

class TritonNativeI4Test : public TritonTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = TritonTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_triton_i4_rewrites(true);
    return debug_options;
  }
};

TEST_F(TritonNativeI4Test, LhsWithMultiply) {
  constexpr absl::string_view kHloText = R"(
    HloModule NativeTritonInt4DotWithMultiply

    gemm_fusion_dot.2_computation {
      w = s4[32,64,128]{2,1,0} parameter(0)
      w.i8 = s8[32,64,128]{2,1,0} convert(w)
      w.f32 = f32[32,64,128]{2,1,0} convert(w.i8)
      scales = f32[32,128]{1,0} parameter(1)
      scales.broadcast = f32[32,64,128]{2,1,0} broadcast(scales), dimensions={0,2}
      weights.scaled = f32[32,64,128]{2,1,0} multiply(w.f32, scales.broadcast)
      activations = f32[32,64,256]{2,1,0} parameter(2)
      ROOT dot.0 = f32[32,128,256]{2,1,0} dot(weights.scaled, activations),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
    } // gemm_fusion_dot.2_computation

    ENTRY gemm_fusion_dot_computation {
      w = s4[32,64,128]{2,1,0} parameter(0)
      scales = f32[32,128]{1,0} parameter(1)
      p2 = f32[32,64,256]{2,1,0} parameter(2)
      ROOT gemm_fusion_dot.2 = f32[32,128,256]{2,1,0} fusion(w, scales, p2),
        kind=kCustom,
        calls=gemm_fusion_dot.2_computation,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config":{
            "kind":"__triton_gemm",
            "triton_gemm_config":{
              "block_m":"128",
              "block_n":"128",
              "block_k":"32",
              "split_k":"1",
              "num_stages":"4",
              "num_warps":"4",
              "num_ctas":"1"
            }
          },
          "force_earliest_schedule":false
        }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

using ::testing::TestParamInfo;
using ::testing::WithParamInterface;

struct I4TestParams {
  static std::string ToString(const TestParamInfo<I4TestParams>& params) {
    return params.param.name;
  }

  std::string Format(absl::string_view format) const {
    return absl::StrReplaceAll(
        format, {{"${name}", name},
                 {"${lhs}", lhs},
                 {"${rhs}", rhs},
                 {"${lhs_contracting_dim}", absl::StrCat(lhs_contracting_dim)},
                 {"${rhs_contracting_dim}", absl::StrCat(rhs_contracting_dim)},
                 {"${out}", out}});
  }

  std::string name;
  std::string lhs;
  std::string rhs;
  int lhs_contracting_dim;
  int rhs_contracting_dim;
  std::string out;
};

class ParametrizedTritonNativeI4Test : public TritonNativeI4Test,
                                       public WithParamInterface<I4TestParams> {
};

TEST_P(ParametrizedTritonNativeI4Test, Lhs) {
  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule ${name}

    NativeTritonInt4Dot.computation {
      w.s4 = s4[${lhs}]{1,0} parameter(0)
      w.s8 = s8[${lhs}]{1,0} convert(w.s4)
      w.f32 = f32[${lhs}]{1,0} convert(w.s8)
      a = f32[${rhs}]{1,0} parameter(1)
      ROOT dot.0 = f32[${out}]{1,0} dot(w.f32, a),
        lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_contracting_dims={${rhs_contracting_dim}}
    } // gemm_fusion_dot.2_computation

    ENTRY gemm_fusion_dot_computation {
      w = s4[${lhs}]{1,0} parameter(0)
      a = f32[${rhs}]{1,0} parameter(1)
      ROOT gemm_fusion_dot.2 = f32[${out}]{1,0} fusion(w, a),
        kind=kCustom,
        calls=NativeTritonInt4Dot.computation,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          },
          "force_earliest_schedule":false
        }
    }
  )";
  std::string kHloText = GetParam().Format(kHloTextTemplate);
  LOG(INFO) << "HLO: " << kHloText;
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_P(ParametrizedTritonNativeI4Test, Rhs) {
  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule ${name}

    NativeTritonInt4Dot.computation {
      a = f32[${lhs}]{1,0} parameter(0)
      w.s4 = s4[${rhs}]{1,0} parameter(1)
      w.s8 = s8[${rhs}]{1,0} convert(w.s4)
      w.f32 = f32[${rhs}]{1,0} convert(w.s8)
      ROOT dot.0 = f32[${out}]{1,0} dot(a, w.f32),
        lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_contracting_dims={${rhs_contracting_dim}}
    } // gemm_fusion_dot.2_computation

    ENTRY gemm_fusion_dot_computation {
      a = f32[${lhs}]{1,0} parameter(0)
      w = s4[${rhs}]{1,0} parameter(1)
      ROOT gemm_fusion_dot.2 = f32[${out}]{1,0} fusion(a, w),
        kind=kCustom,
        calls=NativeTritonInt4Dot.computation,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          },
          "force_earliest_schedule":false
        }
    }
  )";
  std::string kHloText = GetParam().Format(kHloTextTemplate);
  LOG(INFO) << "HLO: " << kHloText;
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

std::vector<I4TestParams> Int4TestCases() {
  return {
      {"dot_128_16_x_128_256", "128,16", "128,256", 0, 0, "16,256"},
      {"dot_128_16_x_256_128", "128,16", "256,128", 0, 1, "16,256"},
      {"dot_16_128_x_256_128", "16,128", "256,128", 1, 1, "16,256"},
      {"dot_16_128_x_128_256", "16,128", "128,256", 1, 0, "16,256"},
      {"dot_1_128_x_256_128", "1,128", "256,128", 1, 1, "1,256"},
      {"dot_128_1_x_256_128", "128,1", "256,128", 0, 1, "1,256"},
      {"dot_16_128_x_128_1", "16,128", "128,1", 1, 0, "16,1"},
      {"dot_16_128_x_1_128", "16,128", "1,128", 1, 1, "16,1"},
  };
}

INSTANTIATE_TEST_SUITE_P(Native, ParametrizedTritonNativeI4Test,
                         ::testing::ValuesIn(Int4TestCases()),
                         I4TestParams::ToString);

TEST_F(TritonTest, NonstandardLayoutInt4) {
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayout

    ENTRY main {
      p0 = s4[64,128]{0,1} parameter(0)
      p1 = bf16[256,64]{1,0} parameter(1)
      ROOT %dot = bf16[128,256]{1,0} dot(s4[64,128]{0,1} p0, bf16[256,64]{1,0} p1),
        lhs_contracting_dims={0},
        rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
           CHECK:  %[[param_0:.*]] = s4[64,128]{0,1:E(4)} parameter(0)
           CHECK:  %[[bitcast:.*]] = s4[128,64]{1,0:E(4)} bitcast(s4[64,128]{0,1:E(4)} %[[param_0]])
           CHECK:  %[[convert:.*]] = bf16[128,64]{1,0} convert(s4[128,64]{1,0:E(4)} %[[bitcast]])
           CHECK:  %[[param_1:.*]] = bf16[256,64]{1,0} parameter(1)
           CHECK:  ROOT %dot.1 = bf16[128,256]{1,0} dot(bf16[128,64]{1,0} %[[convert]], bf16[256,64]{1,0} %[[param_1]]), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  )"));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, NonstandardLayoutWithManyNonContractingDims) {
  // We cannot do triton_gemm and we use cuBLAS instead.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    ENTRY main {
          p0 = s4[128,64,192]{1,0,2} parameter(0)
          p1 = bf16[256,64]{1,0} parameter(1)
          ROOT %dot = bf16[128,192,256]{2,1,0} dot(p0, p1),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(CHECK:  "__cublas$gemm")"));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-2}));
}

TEST_F(TritonTest, NonstandardLayoutWithManyNonContractingDimsReversedLayout) {
  // We cannot do triton_gemm and we use cuBLAS instead.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    ENTRY main {
          p0 = s4[128,64,192]{0,1,2} parameter(0)
          p1 = bf16[256,64]{1,0} parameter(1)
          ROOT %dot = bf16[128,192,256]{2,1,0} dot(p0, p1),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(CHECK:  "__cublas$gemm")"));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, NegatePlusConvertHLO) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

    ENTRY main {
      lhs = s4[16,32,64]{2,1,0} parameter(0)
      lhs_negated = s4[16,32,64]{2,1,0} negate(lhs)
      lhs_converted = bf16[16,32,64]{2,1,0} convert(lhs_negated)
      rhs = bf16[16,64,16]{2,1,0} parameter(1)
      ROOT dot = bf16[16,32,16]{2,1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={2},
          rhs_contracting_dims={1},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, RejectTritonFusionForWithMinorBatchDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

    ENTRY main {
      lhs = s4[32,64,16]{2,1,0} parameter(0)
      lhs_converted = bf16[32,64,16]{2,1,0} convert(lhs)
      rhs = bf16[16,64,16]{2,1,0} parameter(1)
      ROOT dot = bf16[16,32,16]{2,1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1},
          lhs_batch_dims={2},
          rhs_batch_dims={0}
    }
  )";

  const std::string pattern =
      R"(CHECK-NOT: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonTest, LHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = s4[16,1024,1]{2,1,0} parameter(0)
      lhs_converted = bf16[16,1024,1]{2,1,0} convert(lhs)
      rhs = bf16[16,64,1024]{2,1,0} parameter(1)
      ROOT dot = bf16[16,1,64]{2,1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={1},
          rhs_contracting_dims={2},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }

    ENTRY main {
      lhs = s4[16,1024,1]{2,1,0} parameter(0)
      rhs = bf16[16,64,1024]{2,1,0} parameter(1)
      ROOT dot = bf16[16,1,64]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, RHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = bf16[16,1024,64]{2,1,0} parameter(0)
      rhs = s4[16,1024,1]{2,1,0} parameter(1)
      rhs_converted = bf16[16,1024,1]{2,1,0} convert(rhs)
      ROOT dot = bf16[16,64,1]{2,1,0} dot(lhs, rhs_converted),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }

    ENTRY main {
      lhs = bf16[16,1024,64]{2,1,0} parameter(0)
      rhs = s4[16,1024,1]{2,1,0} parameter(1)
      ROOT dot = bf16[16,64,1]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, LHSNonMinorContractingDim) {
  // We prove that triton can handle int4 dot with non minor
  // lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = s4[1024,8]{1,0} parameter(0)
      lhs_converted = bf16[1024,8]{1,0} convert(lhs)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={0},
          rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[1024,8]{1,0} parameter(0)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, LHSNonMinorContractingDimWithBatchDim0) {
  // We prove that triton can handle int4 dot with non minor
  // lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = s4[16,1024,8]{2,1,0} parameter(0)
      lhs_converted = bf16[16,1024,8]{2,1,0} convert(lhs)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} dot(lhs_converted, rhs),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = s4[16,1024,8]{2,1,0} parameter(0)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, LHSMinorContractingDim) {
  // We prove that triton can handle int4 dot with minor lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = s4[8,1024]{1,0} parameter(0)
      lhs_converted = bf16[8,1024]{1,0} convert(lhs)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} dot(lhs_converted, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[8,1024]{1,0} parameter(0)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, ConvertPlusNegate) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = s4[8,1024]{1,0} parameter(0)
      lhs_converted = bf16[8,1024]{1,0} convert(lhs)
      lhs_negated = bf16[8,1024]{1,0} negate(lhs_converted)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} dot(lhs_negated, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[8,1024]{1,0} parameter(0)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, LHSMinorContractingDimWithBatchDim0) {
  // We prove that triton can handle int4 dot with minor lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = s4[16,8,1024]{2,1,0} parameter(0)
      lhs_converted = bf16[16,8,1024]{2,1,0} convert(lhs)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} dot(lhs_converted, rhs),
        lhs_batch_dims={0},
        lhs_contracting_dims={2},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = s4[16,8,1024]{2,1,0} parameter(0)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithNotMinorContractingDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[1024,4]{1,0} parameter(1)
      rhs_converted = bf16[1024,4]{1,0} convert(rhs)
      ROOT dot = bf16[8,4] dot(lhs, rhs_converted),
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithMinorContractingDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[4,1024]{1,0} parameter(1)
      rhs_converted = bf16[4,1024]{1,0} convert(rhs)
      ROOT dot = bf16[8,4] dot(lhs, rhs_converted),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[4,1024]{1,0} parameter(1)
      ROOT dot = bf16[8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithMinorContractingDimWithBatchDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,1024,4]{2,1,0} parameter(1)
      rhs_converted = bf16[16,1024,4]{2,1,0} convert(rhs)
      ROOT dot = bf16[16,8,4] dot(lhs, rhs_converted),
          lhs_batch_dims={0},
          lhs_contracting_dims={2},
          rhs_batch_dims={0},
          rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithNotMinorContractingDimWithBatchDim0) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,4,1024]{2,1,0} parameter(1)
      rhs_converted = bf16[16,4,1024]{2,1,0} convert(rhs)
      ROOT dot = bf16[16,8,4] dot(lhs, rhs_converted),
          lhs_batch_dims={0},
          lhs_contracting_dims={2},
          rhs_batch_dims={0},
          rhs_contracting_dims={2}
    }

    ENTRY main {
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,4,1024]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
