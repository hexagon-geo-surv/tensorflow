
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

#include "tensorflow/lite/experimental/litert/core/model/litert_to_flatbuffer.h"

#include <memory>
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace litert::internal {

namespace {

Expected<TflElementType> MapElementType(LiteRtElementType litert_element_type) {
  switch (litert_element_type) {
    case kLiteRtElementTypeFloat32:
      return tflite::TensorType_FLOAT32;
    case kLiteRtElementTypeFloat16:
      return tflite::TensorType_FLOAT16;
    case kLiteRtElementTypeInt32:
      return tflite::TensorType_INT32;
    case kLiteRtElementTypeBool:
      return tflite::TensorType_BOOL;
    case kLiteRtElementTypeInt16:
      return tflite::TensorType_INT16;
    case kLiteRtElementTypeInt8:
      return tflite::TensorType_INT8;
    default:
      return Error(kLiteRtStatusErrorUnsupported);
  }
}

template <class LiteRtTenzorType>
Expected<TflTensorType> MapTensorTypeDetail(
    const LiteRtTenzorType& litert_tensor_type) {
  return Error(kLiteRtStatusErrorUnsupported);
}

template <>
Expected<TflTensorType> MapTensorTypeDetail<LiteRtRankedTensorType>(
    const LiteRtRankedTensorType& litert_tensor_type) {
  auto tfl_element_type = MapElementType(litert_tensor_type.element_type);
  if (!tfl_element_type) {
    return tfl_element_type.Error();
  }

  auto litert_shape = absl::MakeConstSpan(litert_tensor_type.layout.dimensions,
                                          litert_tensor_type.layout.rank);
  return std::make_pair(*tfl_element_type, TflShapeInfo(litert_shape));
}

Expected<TflQuantizationPtr> MapQuantizationDetail(
    const LiteRtQuantizationParameters& litert_quantization) {
  auto tfl_quantization = std::make_unique<TflQuantization>();
  tfl_quantization->scale.reserve(litert_quantization.scale.size());
  tfl_quantization->zero_point.reserve(litert_quantization.zero_point.size());
  for (const auto& scale : litert_quantization.scale) {
    tfl_quantization->scale.push_back(scale);
  }
  for (const auto& zero_point : litert_quantization.zero_point) {
    tfl_quantization->zero_point.push_back(zero_point);
  }
  return tfl_quantization;
}

}  // namespace

Expected<TflTensorType> MapTensorType(const TensorType& litert_tensor_type) {
  switch (litert_tensor_type.first) {
    case kLiteRtRankedTensorType:
      return MapTensorTypeDetail(litert_tensor_type.second.ranked_tensor_type);
    default:
      return Error(kLiteRtStatusErrorUnsupported);
  }
}

Expected<TflQuantizationPtr> MapQuantization(
    const Quantization& litert_quantization) {
  if (litert_quantization.first == kLiteRtQuantizationNone) {
    return TflQuantizationPtr(nullptr);
  }
  return MapQuantizationDetail(litert_quantization.second);
}

}  // namespace litert::internal
