/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <ios>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {
namespace {

std::vector<char> CreateMultiOpModel() {
  flatbuffers::FlatBufferBuilder builder;
  const std::array<flatbuffers::Offset<OperatorCode>, 2> operator_codes{{
      CreateOperatorCode(builder, BuiltinOperator_ABS),
      CreateOperatorCode(builder, BuiltinOperator_NEG),
  }};

  const std::array<flatbuffers::Offset<Buffer>, 1> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 2> shape{{1, 16}};
  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(shape.data(), shape.size()),
                   TensorType_FLOAT32),
  }};

  const std::array<int32_t, 1> op0_inputs{{0}};
  const std::array<int32_t, 1> op0_outputs{{1}};
  const std::array<int32_t, 1> op1_inputs{{1}};
  const std::array<int32_t, 1> op1_outputs{{2}};
  const std::array<flatbuffers::Offset<Operator>, 2> ops{{
      CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector<int32_t>(op0_inputs.data(), op0_inputs.size()),
          builder.CreateVector<int32_t>(op0_outputs.data(),
                                        op0_outputs.size())),
      CreateOperator(
          builder, /*opcode_index=*/1,
          builder.CreateVector<int32_t>(op1_inputs.data(), op1_inputs.size()),
          builder.CreateVector<int32_t>(op1_outputs.data(),
                                        op1_outputs.size())),
  }};

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{2}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(ops.data(), ops.size()));

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("Multi-op model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace

TEST(Delegate, CreateWithoutParams) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, CreateWithDefaultParams) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, CreateWithNumThreadsParam) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);
}

TEST(Delegate, GetThreadPool) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  pthreadpool_t threadpool = static_cast<pthreadpool_t>(
      TfLiteXNNPackDelegateGetThreadPool(xnnpack_delegate.get()));
  ASSERT_TRUE(threadpool);
  ASSERT_EQ(2, pthreadpool_get_threads_count(threadpool));
}

TEST(Delegate, ConcurrentInvokeAndSubgraphCreateDestroy) {
  std::vector<char> buffer = CreateMultiOpModel();
  const Model* model = GetModel(buffer.data());
  ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::unique_ptr<Interpreter> invoke_interpreter;
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&invoke_interpreter),
            kTfLiteOk);
  ASSERT_NE(invoke_interpreter, nullptr);
  ASSERT_EQ(invoke_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(invoke_interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
            kTfLiteOk);
  ASSERT_EQ(invoke_interpreter->Invoke(), kTfLiteOk);

  std::atomic<bool> stop{false};
  std::thread invoke_thread([&]() {
    int step = 1;
    while (!stop.load(std::memory_order_relaxed)) {
      EXPECT_EQ(invoke_interpreter->ResizeInputTensor(
                    invoke_interpreter->inputs()[0], {1, 16 + step}),
                kTfLiteOk);
      EXPECT_EQ(invoke_interpreter->AllocateTensors(), kTfLiteOk);
      EXPECT_EQ(invoke_interpreter->Invoke(), kTfLiteOk);
      ++step;
    }
  });

  std::thread create_destroy_thread([&]() {
    for (int i = 0; i < 50; ++i) {
      std::unique_ptr<Interpreter> interpreter;
      EXPECT_EQ(InterpreterBuilder(model, resolver)(&interpreter), kTfLiteOk);
      ASSERT_NE(interpreter, nullptr);
      EXPECT_EQ(interpreter->ResizeInputTensor(interpreter->inputs()[0],
                                               {1, 16 * (i + 1)}),
                kTfLiteOk);
      EXPECT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
      EXPECT_EQ(interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
                kTfLiteOk);
      EXPECT_EQ(interpreter->Invoke(), kTfLiteOk);
    }
    stop.store(true, std::memory_order_relaxed);
  });

  create_destroy_thread.join();
  invoke_thread.join();

  std::thread destroy_thread([&]() { invoke_interpreter.reset(); });
  xnnpack_delegate.reset();
  destroy_thread.join();
}

namespace {

std::vector<char> CreateBlockwiseInt4FcModel(
    const std::vector<uint16_t>& fp16_scales) {
  constexpr int32_t kInputChannels = 64;
  constexpr int32_t kOutputChannels = 4;
  constexpr int32_t kBlockSize = 32;
  constexpr int32_t kNumBlocks = kInputChannels / kBlockSize;

  flatbuffers::FlatBufferBuilder builder;
  const std::array<flatbuffers::Offset<OperatorCode>, 1> operator_codes{{
      CreateOperatorCode(builder, BuiltinOperator_FULLY_CONNECTED),
  }};

  // Signed INT4 value +1 packed two nibbles per byte (0x11).
  const std::vector<uint8_t> filter_data(kOutputChannels * kInputChannels / 2,
                                         0x11);
  const uint8_t* scale_bytes =
      reinterpret_cast<const uint8_t*>(fp16_scales.data());
  const std::vector<uint8_t> scale_data(
      scale_bytes, scale_bytes + fp16_scales.size() * sizeof(uint16_t));

  const std::array<flatbuffers::Offset<Buffer>, 3> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder, builder.CreateVector(filter_data)),
      CreateBuffer(builder, builder.CreateVector(scale_data)),
  }};

  const std::array<int32_t, 2> input_shape{{1, kInputChannels}};
  const std::array<int32_t, 2> filter_shape{{kOutputChannels, kInputChannels}};
  const std::array<int32_t, 2> scale_shape{{kOutputChannels, kNumBlocks}};
  const std::array<int32_t, 2> output_shape{{1, kOutputChannels}};

  auto blockwise_details = CreateBlockwiseQuantization(
      builder, /*scales=*/2, /*zero_points=*/-1, /*block_size=*/kBlockSize);
  auto filter_quant = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, blockwise_details.Union(),
      /*quantized_dimension=*/0);

  const std::array<flatbuffers::Offset<Tensor>, 4> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          TensorType_FLOAT32, /*buffer=*/0),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(filter_shape.data(),
                                                 filter_shape.size()),
                   TensorType_INT4, /*buffer=*/1, /*name=*/0, filter_quant),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(scale_shape.data(), scale_shape.size()),
          TensorType_FLOAT16, /*buffer=*/2),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape.data(),
                                                 output_shape.size()),
                   TensorType_FLOAT32, /*buffer=*/0),
  }};

  const std::array<int32_t, 2> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{3}};
  auto fc_options = CreateFullyConnectedOptions(
      builder, ActivationFunctionType_NONE,
      FullyConnectedOptionsWeightsFormat_DEFAULT, /*keep_num_dims=*/false,
      /*asymmetric_quantize_inputs=*/true);
  const std::array<flatbuffers::Offset<Operator>, 1> ops{{
      CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
          builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
          BuiltinOptions_FullyConnectedOptions, fc_options.Union()),
  }};

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{3}};
  auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(ops.data(), ops.size()));

  auto model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("Blockwise INT4 FC model"),
      builder.CreateVector(buffers.data(), buffers.size()));
  builder.Finish(model_buffer);
  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace

TEST(Delegate, BlockwiseInt4ZeroAndSubnormalFp16Scales) {
  // Per-(oc, block) FP16 scales covering zero, subnormals, and normal 1.0.
  const std::vector<uint16_t> fp16_scales = {
      0x0000, 0x0000, 0x0000, 0x3C00, 0x0001, 0x0000, 0x03FF, 0x3C00,
  };
  const std::vector<char> buffer = CreateBlockwiseInt4FcModel(fp16_scales);
  const Model* model = GetModel(buffer.data());
  ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&default_interpreter),
            kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(InterpreterBuilder(model, resolver)(&delegate_interpreter),
            kTfLiteOk);
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);
  ASSERT_EQ(
      delegate_interpreter->ModifyGraphWithDelegate(xnnpack_delegate.get()),
      kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);

  // Verify the single FULLY_CONNECTED node was actually delegated to XNNPACK.
  ASSERT_EQ(delegate_interpreter->execution_plan().size(), 1);
  const auto* node_and_reg = delegate_interpreter->node_and_registration(
      delegate_interpreter->execution_plan()[0]);
  ASSERT_NE(node_and_reg, nullptr);
  EXPECT_EQ(node_and_reg->first.delegate, xnnpack_delegate.get());

  float* default_input = default_interpreter->typed_input_tensor<float>(0);
  float* delegate_input = delegate_interpreter->typed_input_tensor<float>(0);
  for (int i = 0; i < 64; ++i) {
    default_input[i] = 1.0f;
    delegate_input[i] = 1.0f;
  }

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  const float* default_output =
      default_interpreter->typed_output_tensor<float>(0);
  const float* delegate_output =
      delegate_interpreter->typed_output_tensor<float>(0);
  for (int oc = 0; oc < 4; ++oc) {
    EXPECT_NEAR(delegate_output[oc], default_output[oc],
                1e-3f + 1e-2f * std::abs(default_output[oc]))
        << "oc=" << oc;
  }

  // Negative, +Inf, and NaN FP16 scales must still be rejected by XNNPACK.
  for (uint16_t bad_scale :
       {uint16_t{0xBC00}, uint16_t{0x7C00}, uint16_t{0x7E00}}) {
    std::vector<uint16_t> bad_scales(8, 0x3C00);
    bad_scales[1] = bad_scale;
    const std::vector<char> bad_buffer = CreateBlockwiseInt4FcModel(bad_scales);
    const Model* bad_model = GetModel(bad_buffer.data());
    std::unique_ptr<Interpreter> interp;
    ASSERT_EQ(InterpreterBuilder(bad_model, resolver)(&interp), kTfLiteOk);
    const TfLiteStatus modify_status =
        interp->ModifyGraphWithDelegate(xnnpack_delegate.get());
    const TfLiteStatus alloc_status = (modify_status == kTfLiteOk)
                                          ? interp->AllocateTensors()
                                          : modify_status;
    EXPECT_NE(alloc_status, kTfLiteOk)
        << "bad_scale=0x" << std::hex << bad_scale;
  }
}

}  // namespace xnnpack
}  // namespace tflite
