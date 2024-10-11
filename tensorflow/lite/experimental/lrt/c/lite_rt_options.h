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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_OPTIONS_H_

#include <stdbool.h>

#include <cstdint>

#include "tensorflow/lite/schema/schema_generated.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Skipping ReshapeOptions.
// Skipping StableHLOScatterOptions.
typedef tflite::ActivationFunctionType LrtFusedActivationOption;
typedef int32_t LrtAxisOption;
typedef bool LrtAdjXOption;
typedef bool LrtAdjYOption;
typedef bool LrtAsymmetricQuantizeInputOption;
typedef tflite::FullyConnectedOptionsWeightsFormat LrtWeightsFormatOption;
typedef bool LrtKeepNumDimsOption;
typedef tflite::TensorType LrtQuantizedBiasTypeOption;
typedef float LrtBetaOption;
typedef int32_t LrtBeginMaskOption;
typedef int32_t LrtEndMaskOption;
typedef int32_t LrtEllipsisMaskOption;
typedef int32_t LrtNewAxisMaskOption;
typedef int32_t LrtShrinkAxisMaskOption;
typedef bool LrtOffsetOption;
typedef bool LrtPotScaleInt16Option;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_OPTIONS_H_
