// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "xla/tpu/tpu_api.h"
#include "xla/tpu/tpu_profiler_c_api.h"

namespace {

absl::Status SetTpuProfilerApiFns(
    void* library_handle) {  // TENSORFLOW_STATUS_OK
  // Constant cast so that we can initialize the functions. The functions are
  // mutable here because this is the only place where they are initialized.
  auto* profiler_api_fn =
      const_cast<TfTpu_ProfilerApiFn*>(stream_executor::tpu::ProfilerApiFn());
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Create);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Destroy);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Start);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_Stop);
  TFTPU_SET_FN(profiler_api_fn, TpuProfiler_CollectData);

  TFTPU_SET_FN(profiler_api_fn, TpuStatus_New);
  TFTPU_SET_FN(profiler_api_fn, TpuStatus_Free);
  TFTPU_SET_FN(profiler_api_fn, TpuStatus_Message);
  TFTPU_SET_FN(profiler_api_fn, TpuStatus_Code);
  return absl::OkStatus();
}

}  // namespace
