/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_TYPES_H_
#define TENSORFLOW_CORE_PLATFORM_TYPES_H_

#include <cstdint>
#include <limits>
#include <string>

#include "absl/base/macros.h"
#include "xla/tsl/platform/types.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/tstring.h"
#include "tsl/platform/bfloat16.h"
#include "tsl/platform/tstring.h"

namespace tensorflow {

// Alias tensorflow::string to std::string.
using string ABSL_DEPRECATE_AND_INLINE() = std::string;

using tsl::uint2;
using tsl::uint4;
using uint8 ABSL_DEPRECATE_AND_INLINE() = uint8_t;
using uint16 ABSL_DEPRECATE_AND_INLINE() = uint16_t;
using uint32 ABSL_DEPRECATE_AND_INLINE() = uint32_t;
using uint64 ABSL_DEPRECATE_AND_INLINE() = uint64_t;

using tsl::int2;
using tsl::int4;
using int8 ABSL_DEPRECATE_AND_INLINE() = int8_t;
using int16 ABSL_DEPRECATE_AND_INLINE() = int16_t;
using int32 ABSL_DEPRECATE_AND_INLINE() = int32_t;
using int64 ABSL_DEPRECATE_AND_INLINE() = int64_t;

using tsl::float8_e4m3b11fnuz;
using tsl::float8_e4m3fn;
using tsl::float8_e4m3fnuz;
using tsl::float8_e5m2;
using tsl::float8_e5m2fnuz;

inline const uint8_t kuint8max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<uint8_t>::max();
inline const uint16_t kuint16max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<uint16_t>::max();
inline const uint32_t kuint32max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<uint32_t>::max();
inline const uint64_t kuint64max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<uint64_t>::max();
inline const int8_t kint8min ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int8_t>::min();
inline const int8_t kint8max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int8_t>::max();
inline const int16_t kint16min ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int16_t>::min();
inline const int16_t kint16max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int16_t>::max();
inline const int32_t kint32min ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int32_t>::min();
inline const int32_t kint32max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int32_t>::max();
inline const int64_t kint64min ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int64_t>::min();
inline const int64_t kint64max ABSL_DEPRECATE_AND_INLINE() =
    std::numeric_limits<int64_t>::max();

// A typedef for a uint64 used as a short fingerprint.
using tsl::bfloat16;
using tsl::Fprint;
using tsl::tstring;  // NOLINT: suppress 'using decl 'tstring' is unused'
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_TYPES_H_
