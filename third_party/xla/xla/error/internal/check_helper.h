/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_ERROR_INTERNAL_CHECK_HELPER_H_
#define XLA_ERROR_INTERNAL_CHECK_HELPER_H_

#include <ostream>
#include <sstream>
#include <string>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/error/debug_me_context_util.h"

namespace xla::error {

// Helper class to pretty print the debug context and user message when a check
// fails.
class CheckHelper {
 public:
  CheckHelper(const char* absl_nonnull file, int line,
              absl::string_view condition_text)
      : condition_text_(condition_text),
        file_(absl::string_view(file)),
        line_(line) {}

  // Non-copyable/movable.
  CheckHelper(const CheckHelper&) = delete;
  CheckHelper& operator=(const CheckHelper&) = delete;

  // *always* terminates via LOG(FATAL).
  [[noreturn]] ~CheckHelper() {
    std::string user_message = user_stream_.str();

    LOG(FATAL).AtLocation(file_, line_)
        << "Check failed: " << condition_text_ << " " << user_message << "\n"
        << DebugMeContextToErrorMessageString();
  }

  std::ostream& InternalStream() { return user_stream_; }

 private:
  absl::string_view condition_text_;
  absl::string_view file_;
  int line_;
  std::ostringstream user_stream_;
};

}  // namespace xla::error

#endif  // XLA_ERROR_INTERNAL_CHECK_HELPER_H_
