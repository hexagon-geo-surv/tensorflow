/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include <cmath>

#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

static Graph* PTruncatedNormal(int num_batches, int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32_t>().setValues({num_batches, samples_per_batch});

  // Use mean 0 and stdev 1
  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);

  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setRandom();
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(5.0);

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

static Graph* PTruncatedNormal2SD(int num_batches, int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32_t>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(-2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(2.0);

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

static Graph* PTruncatedNormalOneTail(int num_batches, int samples_per_batch) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor shape_t(DT_INT32, TensorShape({2}));
  shape_t.flat<int32_t>().setValues({num_batches, samples_per_batch});

  Tensor means_t(DT_FLOAT, TensorShape({num_batches}));
  means_t.flat<float>().setConstant(0.0);
  Tensor stdevs_t(DT_FLOAT, TensorShape({num_batches}));
  stdevs_t.flat<float>().setConstant(1.0);
  Tensor minvals_t(DT_FLOAT, TensorShape({num_batches}));
  minvals_t.flat<float>().setConstant(2.0);
  Tensor maxvals_t(DT_FLOAT, TensorShape({num_batches}));
  maxvals_t.flat<float>().setConstant(std::numeric_limits<float>::infinity());

  Node* ret;
  TF_CHECK_OK(
      NodeBuilder(g->NewName("truncatednormal"), "ParameterizedTruncatedNormal")
          .Input(test::graph::Constant(g, shape_t))
          .Input(test::graph::Constant(g, means_t))
          .Input(test::graph::Constant(g, stdevs_t))
          .Input(test::graph::Constant(g, minvals_t))
          .Input(test::graph::Constant(g, maxvals_t))
          .Attr("dtype", DT_FLOAT)
          .Finalize(g, &ret));
  return g;
}

#define BM_PTruncatedNormalDev(DEVICE, B, S)                                   \
  static void BM_PTruncatedNormal_##DEVICE##_##B##_##S(                        \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormal(B, S),                           \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_2SD(DEVICE, B, S)                               \
  static void BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S(                    \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormal2SD(B, S),                        \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_2SD_##DEVICE##_##B##_##S);

#define BM_PTruncatedNormalDev_OneTail(DEVICE, B, S)                           \
  static void BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S(                \
      ::testing::benchmark::State& state) {                                    \
    test::Benchmark(#DEVICE, PTruncatedNormalOneTail(B, S),                    \
                    /*old_benchmark_api*/ false)                               \
        .Run(state);                                                           \
    state.SetItemsProcessed(static_cast<int64_t>(B) * S * state.iterations()); \
  }                                                                            \
  BENCHMARK(BM_PTruncatedNormal_OneTail_##DEVICE##_##B##_##S);

BM_PTruncatedNormalDev(cpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(cpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(cpu, 10000, 100);
BM_PTruncatedNormalDev(gpu, 1000, 1000);
BM_PTruncatedNormalDev_2SD(gpu, 10000, 100);
BM_PTruncatedNormalDev_OneTail(gpu, 10000, 100);

class ParameterizedTruncatedNormalOpTest : public OpsTestBase {
 protected:
  void Init(DataType dtype) {
    TF_CHECK_OK(NodeDefBuilder("op", "ParameterizedTruncatedNormal")
                    .Input(FakeInput(DT_INT32))  // shape
                    .Input(FakeInput(dtype))     // means
                    .Input(FakeInput(dtype))     // stddevs
                    .Input(FakeInput(dtype))     // minvals
                    .Input(FakeInput(dtype))     // maxvals
                    .Attr("dtype", dtype)
                    .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(ParameterizedTruncatedNormalOpTest, TestIntegerOverflow) {
  Init(DT_FLOAT);

  // We pass shape that overflows 32-bit: [2, 1073741824]
  // shape: [2, 1073741824], which has 2147483648 elements.
  AddInputFromList<int32_t>(TensorShape({2}), {2, 1073741824});

  // To bypass the scalar batching check, we pass parameters of shape [1]
  // (vector/1-D).
  AddInputFromList<float>(TensorShape({1}), {0.0f});
  AddInputFromList<float>(TensorShape({1}), {1.0f});
  AddInputFromList<float>(TensorShape({1}), {-2.0f});
  AddInputFromList<float>(TensorShape({1}), {2.0f});

  // Runs the op. Must either OOM cleanly or generate valid values.
  // On buggy code, it returns immediately with uninitialized memory.
  absl::Status s = RunOpKernel();
  if (s.code() == error::RESOURCE_EXHAUSTED) {
    // OOM is a valid result.
    return;
  }
  TF_ASSERT_OK(s);

  // Slice to avoid pagefaulting the 8GB tensor.
  Tensor* output = GetOutput(0);
  ASSERT_NE(output, nullptr);
  auto flat_output = output->flat<float>();

  // Check the first 100 elements.
  bool all_zeros = true;
  for (int i = 0; i < 100; ++i) {
    float val = flat_output(i);
    // Values must be within boundary limits.
    EXPECT_GE(val, -2.0f);
    EXPECT_LE(val, 2.0f);
    EXPECT_FALSE(std::isnan(val));
    if (val != 0.0f) {
      all_zeros = false;
    }
  }
  // Check that generation is active (not all values identical/zero).
  EXPECT_FALSE(all_zeros)
      << "Returned uninitialized/zeroed memory instead of generating samples.";
}

}  // namespace tensorflow
