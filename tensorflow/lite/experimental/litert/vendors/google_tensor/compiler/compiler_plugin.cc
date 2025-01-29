// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/adapter.h"

//
// Configurations
//

namespace google_tensor {

constexpr char kPluginManufacturer[] = "GoogleTensor";

constexpr const char* kPluginSocModels[] = {
    "P25",
};  // get the name for plugin soc model

constexpr LiteRtOpCode kSupportedOps[] = {
    kLiteRtOpCodeTflMul,
};
// clang format on

constexpr auto kNumPluginSocModels =
    sizeof(kPluginSocModels) / sizeof(kPluginSocModels[0]);

}  // namespace google_tensor

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return google_tensor::kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = google_tensor::kNumPluginSocModels;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr ||
      soc_model_idx >= google_tensor::kNumPluginSocModels ||
      soc_model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = google_tensor::kPluginSocModels[soc_model_idx];
  return kLiteRtStatusOk;
}

//
// Compiled Result Definition
//

// TODO (abhirs): Revisit this struct after updating the compiler api wrapper to
// return multiple bytecodes.
struct LiteRtCompiledResultT {
  std::string byte_code;
  std::vector<std::string> per_op_data;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (!compiled_result || !byte_code || !byte_code_size ||
      (byte_code_idx != 0)) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *byte_code = compiled_result->byte_code.data();
  *byte_code_size = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (!compiled_result || !call_info || !call_info_size) {
    return kLiteRtStatusErrorInvalidArgument;
  } else if (call_idx >= compiled_result->per_op_data.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  *call_info = compiled_result->per_op_data.at(call_idx).data();
  *call_info_size = compiled_result->per_op_data.at(call_idx).size();
  *byte_code_idx = 0;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (!compiled_result || !num_calls) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls = compiled_result->per_op_data.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

//
// Plugin Definition
//

// Plugins can hold state.
struct LiteRtCompilerPluginT {};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin) {
  *compiler_plugin = new LiteRtCompilerPluginT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  if (compiler_plugin == nullptr) {
    return;
  }
  delete compiler_plugin;
}

namespace google_tensor {
//  TODO(abhirs): update the function to use the darwinn inbuilt way of
//  finding supportedops
bool IsOpSupported(const litert::Op& op) {
  for (auto supported_op : kSupportedOps) {
    if (supported_op == op.Code()) {
      return true;
    }
  }
  return false;
}

}  // namespace google_tensor

LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph graph(subgraph);
  for (const auto& op : graph.Ops()) {
    if (!google_tensor::IsOpSupported(op)) {
      continue;
    }

    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get()));
  }

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (compiler_plugin == nullptr || soc_model == nullptr ||
      partitions == nullptr || compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto num_partitions = partitions->NumSubgraphs();
  auto result = std::make_unique<LiteRtCompiledResultT>();

  // Serialize model.
  litert::OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  LITERT_RETURN_IF_ERROR(
      LiteRtSerializeModel(partitions, &data, &size, &offset));
  // TODO(abhirs): add support for serializing subgraphs

  absl::string_view buffer_str(reinterpret_cast<const char*>(buf.Data()),
                               buf.Size());

  // Compile model.
  auto adapter = litert::google_tensor::Adapter::Create(
      /*shared_library_dir=*/std::nullopt);
  if (!adapter) {
    return adapter.Error().Status();
  }

  // TODO(abhirs): add support for multiple bytecodes
  auto compiled = (*adapter)->api().compile(buffer_str, soc_model);

  if (!compiled.ok()) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  result->byte_code = std::string(compiled->data(), compiled->size());
  // Generate per_op_data.
  for (auto i = 0; i < num_partitions; ++i) {
    char* per_op_data;
    (void)asprintf(&per_op_data, "Partition_%d", i);
    result->per_op_data.push_back(per_op_data);
    free(per_op_data);
  }
  return kLiteRtStatusOk;
}
