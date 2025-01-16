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

#include "xla/python/ifrt/host_callback.h"

#include "absl/strings/string_view.h"

namespace xla {
namespace ifrt {

// TODO(dsuo): Perhaps there's a less round-about way to get an xla::nb_dtype
// from a DataType.
absl::string_view PrimitiveTypeToString(PrimitiveType dtype) {
  switch (dtype) {
    case PrimitiveType::S8:
      return "int8";
    case PrimitiveType::U8:
      return "uint8";
    case PrimitiveType::S16:
      return "int16";
    case PrimitiveType::U16:
      return "uint16";
    case PrimitiveType::S32:
      return "int32";
    case PrimitiveType::U32:
      return "uint32";
    case PrimitiveType::S64:
      return "int64";
    case PrimitiveType::U64:
      return "uint64";
    case PrimitiveType::F16:
      return "float16";
    case PrimitiveType::F32:
      return "float32";
    case PrimitiveType::F64:
      return "float64";
    case PrimitiveType::BF16:
      return "bfloat16";
    case PrimitiveType::C64:
      return "complex64";
    case PrimitiveType::C128:
      return "complex128";
    case PrimitiveType::PRED:
      return "bool";
    case PrimitiveType::TUPLE:
      return "tuple";
    case PrimitiveType::TOKEN:
      return "token";
    default:
      return "unknown";
  }
}

char HostCallback::ID = 0;
char LoadedHostCallback::ID = 0;

}  // namespace ifrt
}  // namespace xla
