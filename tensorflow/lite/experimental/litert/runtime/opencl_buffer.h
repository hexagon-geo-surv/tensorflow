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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_BUFFER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_BUFFER_H_

#include <cstddef>
#include <cstdlib>

#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#if LITERT_HAS_MLDRIFT_SUPPORT
#include "third_party/ml_drift/cl/buffer.h"
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
#include "third_party/opencl_headers/CL/cl.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace internal {

/**
 * The OpenCL buffer class that provides GPU memory allocation and two-way sync
 * between the CPU memory and the GPU OpenCL buffer.
 */
class OpenCLBuffer {
 public:
  OpenCLBuffer(OpenCLBuffer&& other) {
    data_ = other.data_;
#if LITERT_HAS_MLDRIFT_SUPPORT
    buffer_ = std::move(other.buffer_);
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

#if LITERT_HAS_MLDRIFT_SUPPORT
  OpenCLBuffer(ml_drift::cl::Buffer buffer, size_t size)
      : buffer_(std::move(buffer)), size_(size) {}
#endif  // LITERT_HAS_MLDRIFT_SUPPORT

  OpenCLBuffer(cl_mem buffer, size_t size, LiteRtOpenCLDeallocator deallocator)
      : deallocator_(deallocator), size_(size) {
#if LITERT_HAS_MLDRIFT_SUPPORT
    if (deallocator_ != nullptr) {
      buffer_ = ml_drift::cl::CreateBufferShared(buffer);
    } else {  // The buffer will be deallocated automatically.
      buffer_ = ml_drift::cl::Buffer(buffer, size);
    }
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
  }

  ~OpenCLBuffer() {
#if LITERT_HAS_MLDRIFT_SUPPORT
    if (deallocator_ != nullptr) {
      deallocator_(buffer_.GetMemoryPtr());
    }
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
    if (data_ != nullptr) {
      free(data_);
    };
  }

  cl_mem GetMemoryPtr() {
#if LITERT_HAS_MLDRIFT_SUPPORT
    return buffer_.GetMemoryPtr();
#else
    return nullptr;
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
  }
  // Allocates a CPU memory and conducts a copy from the OpenCL buffer to the
  // CPU memory.
  template <typename T>
  Expected<T*> Lock();

  // Writes the data from the CPU memory to the OpenCL buffer.
  template <typename T>
  Expected<void> Unlock();

  static bool IsSupported();
  static Expected<OpenCLBuffer> Alloc(size_t bytes_size);

 private:
  absl::Mutex mutex_;
  // The cpu memory buffer pointer.
  void* data_ = nullptr;
#if LITERT_HAS_MLDRIFT_SUPPORT
  ml_drift::cl::Buffer buffer_;
#endif  // LITERT_HAS_MLDRIFT_SUPPORT
  LiteRtOpenCLDeallocator deallocator_ = nullptr;
  // The size of the buffer in bytes.
  size_t size_ = 0;
};

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_OPENCL_BUFFER_H_
