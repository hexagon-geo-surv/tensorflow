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

#ifndef XLA_FFI_API_RECORD_API_H_
#define XLA_FFI_API_RECORD_API_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/record_c_api.h"

// C++ wrapper for the XLA FFI Record API.
namespace xla::ffi {

// Device pointers are just passed as is by pointer value.
struct DevicePointer {
  void* ptr;
};

// Host values are passed by value to the kernel.
// The size must be non-zero.
// In the end the runtime will copy the value to a temporary buffer.
struct HostValue {
  const void* ptr;
  size_t size;
};

// A kernel argument is either a pointer to a device buffer or a host value.
using KernelArg = std::variant<DevicePointer, HostValue>;

enum class SourceFormat {
  kPtx = XLA_FFI_SourceFormat_PTX,
  kCubin = XLA_FFI_SourceFormat_CUBIN,
};

enum class RecordAction {
  kCreate = XLA_FFI_RecordAction_Create,
  kUpdate = XLA_FFI_RecordAction_Update,
};

namespace internal {

// Wrapper for accessing a pointers to opaque command pointers for recording.
// The container is capped to the maximum number of commands as specified
// during construction.
// This is a just a wrapper to mutate the underlying command vector.
// All arguments to the constructor are non-owning and will be modified in place
// by the wrapper.
template <typename Converter>
class BoundedCommandVector {
 public:
  BoundedCommandVector(const XLA_FFI_Command** commands, int64_t& num_commands,
                       int64_t max_commands)
      : commands_(commands),
        num_commands_(num_commands),
        max_commands_(max_commands) {}
  // Read element by index
  const XLA_FFI_Command* operator[](size_t index) const {
    return commands_[index];
  }

  bool has_storage() const { return commands_ != nullptr; }

  auto push_back(const XLA_FFI_Command* command) {
    if (!has_storage() || num_commands_ >= max_commands_) {
      return Converter::ResourceExhaustedError("CommandVector overflow");
    }
    commands_[num_commands_++] = command;
    return Converter::Success();
  }
  int64_t size() const { return num_commands_; }
  int64_t capacity() const { return max_commands_; }

 private:
  const XLA_FFI_Command** commands_;
  int64_t& num_commands_;
  const int64_t max_commands_;
};

// Note: Cannot use absl::Overload here because it's not guaranteed that absl
// is available in the external FFI module.
template <class... Ts>
struct Overload : Ts... {
  using Ts::operator()...;
};
// Deduction guide.
template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

// C++ wrapper for the XLA FFI Record extension API.
// Unified implementation for statically linked and dynamically linked FFI
// modules.
template <typename Converter>
class RecordContextBase {
 private:
  // Converts a span of `KernelArg` or `void*` to a vector of
  // `XLA_FFI_KernelArg`.
  template <typename KernelArgSpan>
  std::vector<XLA_FFI_KernelArg> ConvertArgs(KernelArgSpan args) {
    std::vector<XLA_FFI_KernelArg> raw_args;
    const size_t num_args = std::size(args);
    raw_args.reserve(num_args);
    for (const auto& arg : args) {
      // Can't use std/absl::remove_cvref because we don't have absl OR cpp20.
      // NOLINTNEXTLINE(modernize-type-traits)
      using ArgT = std::remove_cv_t<std::remove_reference_t<decltype(arg)>>;
      if constexpr (std::is_same_v<ArgT, KernelArg>) {
        std::visit(
            Overload{[&](const DevicePointer& arg) {
                       raw_args.push_back(
                           {arg.ptr, 0, XLA_FFI_KernelArgType_DevicePtr});
                     },
                     [&](const HostValue& arg) {
                       raw_args.push_back({arg.ptr,
                                           static_cast<int64_t>(arg.size),
                                           XLA_FFI_KernelArgType_HostValue});
                     }},
            arg);
      } else {
        // Handle void* arguments for convenience.
        raw_args.push_back({arg, 0, XLA_FFI_KernelArgType_DevicePtr});
      }
    }
    return raw_args;
  }

 public:
  using ReturnType = decltype(Converter::ToStatusOr(
      std::declval<const XLA_FFI_Command*>(), std::declval<XLA_FFI_Error*>()));
  // The constructor takes the main API and the frame directly. (The main API
  // is passed by CtxDecoding but can be ignored since the frame provides the
  // RecordApi).
  explicit RecordContextBase(const XLA_FFI_RecordFrame* frame)
      : frame_(frame) {}

  RecordAction action() const {
    return static_cast<RecordAction>(frame_->action);
  }

  // Instantiate a used facing comamnd wrapper for convenience.
  BoundedCommandVector<Converter> commands() const {
    return BoundedCommandVector<Converter>(
        frame_->commands, *frame_->num_commands, frame_->max_commands);
  }

  // Unified CreateLaunch (handles both Span<const KernelArg> and Span<const
  // void* const> for args)
  template <typename KernelArgSpan,
            typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  ReturnType CreateLaunch(const char* kernel_name, const void* kernel_data,
                          size_t kernel_size, SourceFormat format,
                          XLA_FFI_LaunchDims launch_dims,
                          uint32_t shared_mem_bytes, KernelArgSpan args,
                          DepSpan dependencies = {}) {
    std::vector<XLA_FFI_KernelArg> raw_args = ConvertArgs(args);
    XLA_FFI_KernelArgs ffi_args{raw_args.data(),
                                static_cast<int64_t>(raw_args.size())};
    const XLA_FFI_Command* out_command = nullptr;

    XLA_FFI_Error* err = frame_->api->create_launch(
        frame_->record_ctx, kernel_name, kernel_data, kernel_size,
        static_cast<XLA_FFI_SourceFormat>(format), launch_dims,
        shared_mem_bytes, &ffi_args, std::data(dependencies),
        std::size(dependencies), &out_command);

    auto cmds = commands();
    if (!err && out_command && cmds.has_storage()) {
      auto push_status = cmds.push_back(out_command);
      if (Converter::IsError(push_status)) {
        return ReturnType(push_status);  // Return type correctly handles status
      }
    }
    return Converter::ToStatusOr(out_command, err);
  }

  template <typename KernelArgSpan>
  auto UpdateLaunch(const XLA_FFI_Command* command, KernelArgSpan args) {
    std::vector<XLA_FFI_KernelArg> raw_args = ConvertArgs(args);
    XLA_FFI_KernelArgs ffi_args{raw_args.data(),
                                static_cast<int64_t>(raw_args.size())};
    XLA_FFI_Error* err =
        frame_->api->update_launch(frame_->record_ctx, command, &ffi_args);
    return Converter::ToStatus(err);
  }

  template <typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  auto CreateEmptyCommand(DepSpan dependencies = {}) {
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err = frame_->api->create_empty_command(
        frame_->record_ctx, std::data(dependencies), std::size(dependencies),
        &out_command);

    auto cmds = commands();
    if (!err && out_command && cmds.has_storage()) {
      auto push_status = cmds.push_back(out_command);
      if (Converter::IsError(push_status)) {
        return ReturnType(push_status);
      }
    }
    return Converter::ToStatusOr(out_command, err);
  }

  auto RequestStreamCapture() {
    XLA_FFI_Error* err =
        frame_->api->request_stream_capture(frame_->record_ctx);
    return Converter::ToStatus(err);
  }

  template <typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  auto CreateMemcpyD2D(void* dst, void* src, size_t size,
                       DepSpan dependencies = {}) {
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err = frame_->api->create_memcpy_d2d(
        frame_->record_ctx, dst, src, size, std::data(dependencies),
        std::size(dependencies), &out_command);

    auto cmds = commands();
    if (!err && out_command && cmds.has_storage()) {
      auto push_status = cmds.push_back(out_command);
      if (Converter::IsError(push_status)) {
        return ReturnType(push_status);
      }
    }
    return Converter::ToStatusOr(out_command, err);
  }

  auto UpdateMemcpyD2D(const XLA_FFI_Command* command, void* dst, void* src,
                       size_t size) {
    XLA_FFI_Error* err = frame_->api->update_memcpy_d2d(
        frame_->record_ctx, command, dst, src, size);
    return Converter::ToStatus(err);
  }

 private:
  const XLA_FFI_RecordFrame* frame_;
};

// Common base struct for static and dynamic Record extensions.
// Defines traits for CtxDecoding<Extension<RecordExtension>>.
template <typename ContextType>
struct RecordExtensionBase {
  using Context = ContextType;
  using CExtension = XLA_FFI_Record_Extension;

  static constexpr auto kName = "RecordExtension";
  static constexpr int32_t kExtensionType = XLA_FFI_Extension_RecordFrame;
  static constexpr int32_t kMajorVersion =
      XLA_FFI_Extension_RecordFrame_MajorVersion;
  static constexpr int32_t kMinorVersion =
      XLA_FFI_Extension_RecordFrame_MinorVersion;

  // Builds a context from the extension.
  static Context Instantiate(const CExtension* ext) {
    return Context(ext->record_frame);
  }
};

}  // namespace internal

inline XLA_FFI_Record_Extension BuildRecordCExtension(
    XLA_FFI_RecordFrame* frame) {
  XLA_FFI_Record_Extension ext;
  // context type is not important for constructing the extension since we are
  // not decoding the context here.
  ext.extension_base =
      MakeExtensionHeader<internal::RecordExtensionBase<void>>();
  ext.record_frame = frame;
  return ext;
}

}  // namespace xla::ffi

#endif  // XLA_FFI_API_RECORD_API_H_
