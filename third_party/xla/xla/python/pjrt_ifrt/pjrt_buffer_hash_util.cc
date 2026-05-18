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

#include "xla/python/pjrt_ifrt/pjrt_buffer_hash_util.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/primitive_util.h"
#include "xla/python/ifrt/buffer_hash_util.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace ifrt {

namespace {

// Shared state for all InflightHashing objects.
struct State {
  absl::Mutex mu;
  // Number of buffers that are currently being hashed.
  int inflight_buffers ABSL_GUARDED_BY(mu) = 0;
  // Total size of all buffers that are currently being hashed.
  int64_t current_in_flight_byte_size ABSL_GUARDED_BY(mu) = 0;
  // Accumulated status so far.
  absl::Status status ABSL_GUARDED_BY(mu);
};

// RAII object that updates the shared state when destroyed.
class InflightHashing {
 public:
  InflightHashing(State& state, int64_t byte_size, uint64_t* out_hash)
      : state_(state),
        byte_size_(byte_size),
        out_hash_(out_hash),
        status_(std::nullopt) {}

  ~InflightHashing() {
    CHECK(status_.has_value())
        << "InflightHashing destroyed without setting result or error";
    absl::MutexLock lock(state_.mu);
    state_.status.Update(*status_);
    --state_.inflight_buffers;
    state_.current_in_flight_byte_size -= byte_size_;
  }

  void SetResult(absl::StatusOr<uint64_t> result) {
    if (result.ok()) {
      *out_hash_ = *result;
      status_ = absl::OkStatus();
    } else {
      status_ = result.status();
    }
  }

 private:
  State& state_;
  const int64_t byte_size_;
  uint64_t* out_hash_;
  std::optional<absl::Status> status_;
};

}  // namespace

absl::StatusOr<std::vector<uint64_t>> HashPjRtBuffers(
    absl::Span<PjRtBuffer* const> pjrt_buffers,
    absl::Span<const IndexDomain> index_domains, Client::HashMode mode,
    int64_t max_inflight_memory) {
  if (pjrt_buffers.size() != index_domains.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "buffers and index_domains must have the same size, but have ",
        pjrt_buffers.size(), " vs. ", index_domains.size()));
  }

  if (pjrt_buffers.empty()) {
    return std::vector<uint64_t>();
  }

  std::vector<int64_t> element_byte_sizes;
  std::vector<int64_t> buffer_byte_sizes;
  element_byte_sizes.reserve(pjrt_buffers.size());
  buffer_byte_sizes.reserve(pjrt_buffers.size());
  for (int i = 0; i < pjrt_buffers.size(); ++i) {
    // Use an unpacked element type for sub-byte types.
    const int64_t element_byte_size = xla::primitive_util::ByteWidth(
        pjrt_buffers[i]->on_device_shape().element_type());
    element_byte_sizes.push_back(element_byte_size);
    buffer_byte_sizes.push_back(element_byte_size *
                                index_domains[i].shape().num_elements());
  }

  std::vector<uint64_t> hashes(pjrt_buffers.size());
  State state;

  for (int i = 0; i < pjrt_buffers.size(); ++i) {
    const int64_t buffer_byte_size = buffer_byte_sizes[i];
    {
      absl::MutexLock lock(state.mu);
      auto condition = [&]() ABSL_SHARED_LOCKS_REQUIRED(state.mu) {
        // If a single buffer is larger than the limit, we allow it to exceed
        // the limit, but only if there is no other inflight hasing.
        if (buffer_byte_size > max_inflight_memory) {
          return state.current_in_flight_byte_size == 0;
        }
        return state.current_in_flight_byte_size + buffer_byte_size <=
               max_inflight_memory;
      };
      state.mu.Await(absl::Condition(&condition));
      state.current_in_flight_byte_size += buffer_byte_size;
      ++state.inflight_buffers;
    }

    auto inflight_hashing = std::make_unique<InflightHashing>(
        state, buffer_byte_sizes[i], &hashes[i]);

    const xla::Shape& device_shape = pjrt_buffers[i]->on_device_shape();
    xla::Shape descending_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
        device_shape.element_type(), device_shape.dimensions());
    absl::StatusOr<std::unique_ptr<xla::Literal>> literal =
        xla::Literal::MakeUnique(descending_shape);
    if (!literal.ok()) {
      inflight_hashing->SetResult(literal.status());
      break;
    }

    pjrt_buffers[i]
        ->ToLiteral((*literal).get())
        .OnReady([&, i, literal = *std::move(literal),
                  inflight_hashing = std::move(inflight_hashing)](
                     absl::Status status) mutable {
          if (!status.ok()) {
            inflight_hashing->SetResult(status);
            return;
          }
          // literal is moved into SchedClosure lambda to extend its lifetime.
          tsl::Env::Default()->SchedClosure([&, i, literal = std::move(literal),
                                             inflight_hashing = std::move(
                                                 inflight_hashing)]() mutable {
            absl::StatusOr<uint64_t> hash = [&]() -> absl::StatusOr<uint64_t> {
              absl::Span<const char> literal_span(
                  static_cast<const char*>(literal->untyped_data()),
                  literal->size_bytes());

              std::shared_ptr<const PjRtLayout> pjrt_layout =
                  pjrt_buffers[i]->layout();
              const xla::Layout& xla_layout = pjrt_layout->xla_layout();

              switch (mode) {
                case Client::HashMode::kPhysical:
                  return HashBufferPhysical(literal_span, xla_layout);
                case Client::HashMode::kLogical:
                  return HashBufferLogical(literal_span, element_byte_sizes[i],
                                           index_domains[i]);
              }
            }();
            inflight_hashing->SetResult(std::move(hash));
          });
        });
  }

  {
    auto condition = [&]() ABSL_SHARED_LOCKS_REQUIRED(state.mu) {
      return state.inflight_buffers == 0;
    };
    absl::MutexLock lock(state.mu);
    state.mu.Await(absl::Condition(&condition));
    if (!state.status.ok()) {
      return state.status;
    }
  }
  return hashes;
}

}  // namespace ifrt
}  // namespace xla
