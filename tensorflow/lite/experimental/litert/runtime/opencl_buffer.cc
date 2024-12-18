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

#include "tensorflow/lite/experimental/litert/runtime/opencl_buffer.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#if LITERT_HAS_MLDRIFT_SUPPORT
#include "third_party/ml_drift/cl/buffer.h"
#include "third_party/ml_drift/cl/cl_command_queue.h"
#include "third_party/ml_drift/cl/cl_context.h"
#include "third_party/ml_drift/cl/environment.h"
#include "third_party/ml_drift/cl/opencl_wrapper.h"
#include "third_party/ml_drift/cl/util_types.h"
#endif  // LITERT_HAS_MLDRIFT_SUPPORT

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace internal {
#if LITERT_HAS_MLDRIFT_SUPPORT
// Inner singleton class that is for storing the MLD OpenCL environment.
// TODO(piyu): MLD CL environment will need to be per model configuration.
class EnvironmentSingleton {
 public:
  EnvironmentSingleton(const EnvironmentSingleton&) = delete;
  EnvironmentSingleton& operator=(const EnvironmentSingleton&) = delete;
  ~EnvironmentSingleton() = default;
  ml_drift::cl::Environment* getEnvironment() { return &env_; }

  static EnvironmentSingleton& GetInstance() {
    static EnvironmentSingleton* instance = new EnvironmentSingleton();
    return *instance;
  }

 private:
  EnvironmentSingleton() {
    // TODO(piyu): Add litert environment for opencl options.
    ml_drift::cl::EnvironmentOptions options;
    options.performance = ml_drift::cl::PerformanceHint::kHigh;
    options.priority = ml_drift::cl::PriorityHint::kNormal;
    auto status = ml_drift::cl::CreateEnvironment(&env_, options);
    if (!status.ok()) {
      LITERT_LOG(LITERT_ERROR, "Failed to create OpenCL environment: %s",
                 status.message().data());
    }
  }
  ml_drift::cl::Environment env_;
};
#endif  // LITERT_HAS_MLDRIFT_SUPPORT

template Expected<float*> OpenCLBuffer::Lock<float>();
template Expected<void> OpenCLBuffer::Unlock<float>();

template <typename T>
Expected<T*> OpenCLBuffer::Lock() {
#if LITERT_HAS_MLDRIFT_SUPPORT
  absl::MutexLock lock(&mutex_);
  // The buffer has not been locked, so we need to read from the OpenCL
  // buffer.
  if (data_ == nullptr) {
    ml_drift::cl::CLCommandQueue* queue =
        EnvironmentSingleton::GetInstance().getEnvironment()->queue();
    std::vector<T> result;
    auto status = buffer_.ReadData(queue, &result);
    if (!status.ok()) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to read OpenCL buffer");
    }
    // Ensure the data is aligned.
    if (auto rc = ::posix_memalign(&data_, LITERT_HOST_MEMORY_BUFFER_ALIGNMENT,
                                   size_);
        rc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate aligned memory");
    }
    // Copy the data from the OpenCL buffer to the aligned memory.
    // TODO(piyu): Consider adding support in MLD OpenCL buffer to directly
    // write to the aligned memory.
    std::copy(result.begin(), result.end(), static_cast<T*>(data_));
  }
  return Expected<T*>(static_cast<T*>(data_));
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "OpenCL buffer is not supported");
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
}

template <typename T>
Expected<void> OpenCLBuffer::Unlock() {
#if LITERT_HAS_MLDRIFT_SUPPORT
  absl::MutexLock lock(&mutex_);
  ml_drift::cl::CLCommandQueue* queue =
      EnvironmentSingleton::GetInstance().getEnvironment()->queue();
  // The buffer has not been locked, so we don't need to write back.
  if (data_ == nullptr) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "Cannot unlock a buffer that wasn't locked in the first place");
  }
  size_t write_size = (size_ + sizeof(T) - 1) / sizeof(T);
  auto status = buffer_.WriteData(
      queue, absl::MakeSpan(static_cast<T*>(data_), write_size));

  if (status.ok()) {
    return Expected<void>();
  }
  return Unexpected(
      kLiteRtStatusErrorRuntimeFailure,
      "The data failed to write to the OpenCL buffer when unlocked");
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "OpenCL buffer is not supported");
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
}

bool OpenCLBuffer::IsSupported() {
#if LITERT_HAS_MLDRIFT_SUPPORT
  static bool is_supported = ::ml_drift::cl::LoadOpenCL().ok();
  return is_supported;
#else
  return false;
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
}

Expected<OpenCLBuffer> OpenCLBuffer::Alloc(size_t bytes_size) {
#if LITERT_HAS_MLDRIFT_SUPPORT
  ml_drift::cl::Buffer buffer;

  ml_drift::cl::CLContext& cl_context =
      EnvironmentSingleton::GetInstance().getEnvironment()->context();
  auto result =
      ml_drift::cl::CreateReadWriteBuffer(bytes_size, &cl_context, &buffer);
  if (!result.ok()) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create OpenCL buffer");
  }

  return Expected<OpenCLBuffer>(std::move(buffer), bytes_size);
#else
  return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                    "OpenCL buffer is not supported");
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
}
}  // namespace internal
}  // namespace litert
