/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/qsa_delegate_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "fp16.h"  // from @FP16
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace xnnpack {

class QsaDelegateKernel::Impl {
 public:
  virtual ~Impl() = default;
  virtual TfLiteStatus Prepare(TfLiteContext* context) = 0;
  virtual TfLiteStatus Invoke(TfLiteContext* context) = 0;
};

namespace {

constexpr char kSparseSdpaName[] = "odml.sparse_sdpa_transposed";
constexpr char kIndexerUpdateName[] = "odml.qsa_indexer_update";

// Number of KV tokens processed together by the attention inner loop.
constexpr int kTokenTile = 16;
// Minimum number of kept KV tokens per work item when one query row is split
// across threads (flash-decoding style).
constexpr int kMinTokensPerChunk = 64;
// Minimum number of complete blocks scored per work item when one indexer row
// is split across threads.
constexpr int kMinBlocksPerChunk = 128;

// ---------------------------------------------------------------------------
// fp16 <-> fp32 conversion. fp16 tensors are stored as IEEE binary16 bits.
// ---------------------------------------------------------------------------

inline float ToFloat(float v) { return v; }
inline float ToFloat(uint16_t v) { return fp16_ieee_to_fp32_value(v); }

template <typename T>
inline T FromFloat(float v);
template <>
inline float FromFloat<float>(float v) {
  return v;
}
template <>
inline uint16_t FromFloat<uint16_t>(float v) {
  return fp16_ieee_from_fp32_value(v);
}

// Rounds `v` to the precision of `type` (identity for fp32).
inline float RoundTo(TfLiteType type, float v) {
  return type == kTfLiteFloat16
             ? fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(v))
             : v;
}

template <typename T>
inline void ConvertToFloat(const T* src, float* dst, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = ToFloat(src[i]);
  }
}

// Dot product with independent per-lane partial sums so that the compiler can
// vectorize it without reassociating floating-point additions.
inline float Dot(const float* a, const float* b, int n) {
  constexpr int kLanes = 16;
  float lanes[kLanes] = {};
  int i = 0;
  for (; i + kLanes <= n; i += kLanes) {
    for (int l = 0; l < kLanes; ++l) {
      lanes[l] += a[i + l] * b[i + l];
    }
  }
  float sum = 0.0f;
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  for (int l = 0; l < kLanes; ++l) {
    sum += lanes[l];
  }
  return sum;
}

// Reads element `i` of an fp32 / fp16 tensor.
inline float ReadFloat(const TfLiteTensor& t, size_t i) {
  if (t.type == kTfLiteFloat16) {
    return fp16_ieee_to_fp32_value(
        reinterpret_cast<const uint16_t*>(t.data.raw_const)[i]);
  }
  return reinterpret_cast<const float*>(t.data.raw_const)[i];
}

inline bool IsFloatType(TfLiteType type) {
  return type == kTfLiteFloat32 || type == kTfLiteFloat16;
}

inline bool IsMaskType(TfLiteType type) {
  return type == kTfLiteBool || IsFloatType(type);
}

inline int Dim(const TfLiteTensor& t, int i) { return t.dims->data[i]; }

inline int NumDims(const TfLiteTensor& t) {
  return t.dims == nullptr ? -1 : t.dims->size;
}

inline size_t NumElements(const TfLiteTensor& t) {
  size_t n = 1;
  for (int i = 0; i < t.dims->size; ++i) {
    n *= static_cast<size_t>(t.dims->data[i]);
  }
  return n;
}

// Returns true if `id` refers to a dense tensor with positive dimensions and
// fewer than 2^31 elements, so that int arithmetic on its sizes cannot
// overflow.
bool IsValidTensor(const TfLiteContext* context, int id) {
  if (id < 0 || id >= static_cast<int>(context->tensors_size)) {
    return false;
  }
  const TfLiteTensor& t = context->tensors[id];
  if (t.dims == nullptr || t.sparsity != nullptr) {
    return false;
  }
  int64_t n = 1;
  for (int i = 0; i < t.dims->size; ++i) {
    if (t.dims->data[i] <= 0) {
      return false;
    }
    n *= t.dims->data[i];
    if (n > std::numeric_limits<int32_t>::max()) {
      return false;
    }
  }
  return true;
}

// Returns true if all node inputs are valid tensors and all node outputs are
// in range.
bool HasValidTensors(const TfLiteContext* context, const TfLiteNode* node) {
  for (int i = 0; i < node->inputs->size; ++i) {
    if (!IsValidTensor(context, node->inputs->data[i])) {
      return false;
    }
  }
  for (int i = 0; i < node->outputs->size; ++i) {
    const int id = node->outputs->data[i];
    if (id < 0 || id >= static_cast<int>(context->tensors_size)) {
      return false;
    }
  }
  return true;
}

// Runs `fn(thread_index, item)` for `item` in `[0, count)`.
template <typename Fn>
void ParallelForWithThread(pthreadpool_t threadpool, size_t count, Fn&& fn) {
  if (count == 0) {
    return;
  }
  if (threadpool == nullptr || count == 1) {
    for (size_t i = 0; i < count; ++i) {
      fn(0, i);
    }
    return;
  }
  using FnType = std::decay_t<Fn>;
  pthreadpool_parallelize_1d_with_thread(
      threadpool,
      [](void* ctx, size_t thread, size_t item) {
        (*static_cast<FnType*>(ctx))(thread, item);
      },
      &fn, count, /*flags=*/0);
}

size_t NumThreads(pthreadpool_t threadpool) {
  return threadpool == nullptr
             ? 1
             : std::max<size_t>(1, pthreadpool_get_threads_count(threadpool));
}

// ---------------------------------------------------------------------------
// Attributes.
// ---------------------------------------------------------------------------

const TfLiteStablehloCompositeParams* GetCompositeParams(
    const TfLiteRegistration* registration, const TfLiteNode* node) {
  if (registration == nullptr || node == nullptr ||
      registration->builtin_code != kTfLiteBuiltinStablehloComposite ||
      node->builtin_data == nullptr) {
    return nullptr;
  }
  const auto* params =
      static_cast<const TfLiteStablehloCompositeParams*>(node->builtin_data);
  if (params->name == nullptr) {
    return nullptr;
  }
  return params;
}

struct QsaAttributes {
  int compress_ratio = 0;
  int block_topk = 0;
  float scale = 0.0f;
  bool has_scale = false;
  float rms_norm_eps = 1e-6f;
  int k_ts_idx = 2;
  int v_ts_idx = 3;
};

bool ReadAttributes(TfLiteContext* context,
                    const TfLiteStablehloCompositeParams* params,
                    int node_index, QsaAttributes* attr) {
  if (params->attributes == nullptr || params->attributes_size == 0) {
    TF_LITE_KERNEL_LOG(context, "%s node #%d has no attributes", params->name,
                       node_index);
    return false;
  }
  if (!flexbuffers::VerifyBuffer(params->attributes, params->attributes_size)) {
    TF_LITE_KERNEL_LOG(context, "%s node #%d has malformed attributes",
                       params->name, node_index);
    return false;
  }
  const flexbuffers::Reference root =
      flexbuffers::GetRoot(params->attributes, params->attributes_size);
  if (!root.IsMap()) {
    TF_LITE_KERNEL_LOG(context, "%s node #%d attributes are not a map",
                       params->name, node_index);
    return false;
  }
  const flexbuffers::Map map = root.AsMap();
  // Reads an integer attribute in [lo, hi]; missing keys keep `*value`.
  auto read_int = [&](const char* key, int64_t lo, int64_t hi, bool required,
                      int* value) {
    const flexbuffers::Reference ref = map[key];
    if (ref.IsNull()) {
      return !required;
    }
    const int64_t v = ref.AsInt64();
    if (v < lo || v > hi) {
      return false;
    }
    *value = static_cast<int>(v);
    return true;
  };
  // Reads a finite float attribute in [lo, +inf); missing keys keep `*value`.
  auto read_float = [&](const char* key, float lo, float* value) {
    const flexbuffers::Reference ref = map[key];
    if (ref.IsNull()) {
      return true;
    }
    const double v = ref.AsDouble();
    if (!std::isfinite(v) || v < lo || v > std::numeric_limits<float>::max()) {
      return false;
    }
    *value = static_cast<float>(v);
    return true;
  };
  constexpr int64_t kMaxInt = std::numeric_limits<int32_t>::max();
  attr->has_scale = !map["scale"].IsNull();
  if (!read_int("compress_ratio", 1, kMaxInt, /*required=*/true,
                &attr->compress_ratio) ||
      !read_int("block_topk", 1, kMaxInt, /*required=*/true,
                &attr->block_topk) ||
      !read_int("k_ts_idx", 2, 3, /*required=*/false, &attr->k_ts_idx) ||
      !read_int("v_ts_idx", 2, 3, /*required=*/false, &attr->v_ts_idx) ||
      !read_float("scale", -std::numeric_limits<float>::max(), &attr->scale) ||
      !read_float("rms_norm_eps", 0.0f, &attr->rms_norm_eps)) {
    TF_LITE_KERNEL_LOG(context, "%s node #%d has missing or invalid attributes",
                       params->name, node_index);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Attention mask access.
// ---------------------------------------------------------------------------

// `attention_mask` is `[B_m, H_m, S_m, T]`; only head 0 is used, and B_m / S_m
// may be 1 (broadcast). A FLOAT mask entry is visible iff it is > -1.
class MaskView {
 public:
  explicit MaskView(const TfLiteTensor& mask)
      : mask_(mask),
        batch_(Dim(mask, 0)),
        heads_(Dim(mask, 1)),
        rows_(Dim(mask, 2)),
        cols_(Dim(mask, 3)) {}

  static bool IsCompatible(const TfLiteTensor& mask, int batch, int rows,
                           int cols) {
    return IsMaskType(mask.type) && NumDims(mask) == 4 &&
           (Dim(mask, 0) == batch || Dim(mask, 0) == 1) && Dim(mask, 1) >= 1 &&
           (Dim(mask, 2) == rows || Dim(mask, 2) == 1) && Dim(mask, 3) == cols;
  }

  // Returns a pointer to `cols` visibility bytes for query row (b, s). For
  // non-BOOL masks the row is materialized into `scratch`.
  const uint8_t* Row(int b, int s, std::vector<uint8_t>* scratch) const {
    const size_t bm = batch_ == 1 ? 0 : b;
    const size_t sm = rows_ == 1 ? 0 : s;
    const size_t offset = ((bm * heads_) * rows_ + sm) * cols_;
    if (mask_.type == kTfLiteBool) {
      return reinterpret_cast<const uint8_t*>(mask_.data.raw_const) + offset;
    }
    scratch->resize(cols_);
    uint8_t* out = scratch->data();
    if (mask_.type == kTfLiteFloat32) {
      const float* row =
          reinterpret_cast<const float*>(mask_.data.raw_const) + offset;
      for (int t = 0; t < cols_; ++t) {
        out[t] = row[t] > -1.0f ? 1 : 0;
      }
    } else {
      const uint16_t* row =
          reinterpret_cast<const uint16_t*>(mask_.data.raw_const) + offset;
      for (int t = 0; t < cols_; ++t) {
        out[t] = fp16_ieee_to_fp32_value(row[t]) > -1.0f ? 1 : 0;
      }
    }
    return out;
  }

  // Visibility of a single entry.
  bool At(int b, int s, int t) const {
    const size_t bm = batch_ == 1 ? 0 : b;
    const size_t sm = rows_ == 1 ? 0 : s;
    const size_t i = ((bm * heads_) * rows_ + sm) * cols_ + t;
    switch (mask_.type) {
      case kTfLiteBool:
        return reinterpret_cast<const uint8_t*>(mask_.data.raw_const)[i] != 0;
      case kTfLiteFloat32:
        return reinterpret_cast<const float*>(mask_.data.raw_const)[i] > -1.0f;
      default:
        return fp16_ieee_to_fp32_value(reinterpret_cast<const uint16_t*>(
                   mask_.data.raw_const)[i]) > -1.0f;
    }
  }

 private:
  const TfLiteTensor& mask_;
  int batch_;
  int heads_;
  int rows_;
  int cols_;
};

// Returns true if all `ratio` tokens of the block starting at `base` are
// visible; `*any` is set if at least one token is visible.
inline bool BlockVisibility(const uint8_t* vis, int base, int ratio,
                            bool* any) {
  bool all = true;
  bool some = false;
  for (int r = 0; r < ratio; ++r) {
    const bool v = vis[base + r] != 0;
    all = all && v;
    some = some || v;
  }
  *any = some;
  return all;
}

// ---------------------------------------------------------------------------
// odml.sparse_sdpa_transposed
// ---------------------------------------------------------------------------

struct Segment {
  int32_t begin;
  int32_t end;
};

// Online-softmax accumulator state for `G` query heads.
struct SoftmaxState {
  float* m;    // [G] running max.
  float* l;    // [G] running sum.
  float* acc;  // [G, D] running weighted sum of values.
};

struct SdpaScratch {
  std::vector<uint8_t> vis;
  std::vector<uint8_t> selected;
  std::vector<Segment> segments;
  std::vector<float> q;          // [G, D], pre-scaled.
  std::vector<float> k;          // [kTokenTile, D].
  std::vector<float> v;          // [max(G, D)].
  std::vector<float> logits;     // [G, num_tokens] softmax numerators.
  std::vector<uint8_t> v_raw;    // Gathered value row (raw KV type).
  std::vector<float> v_row;      // Gathered value row in fp32.
  std::vector<float> max_logit;  // [G].
  std::vector<float> state;      // m [G], l [G], acc [G, D].
};

class SparseSdpaKernel : public QsaDelegateKernel::Impl {
 public:
  static TfLiteStatus IsSupported(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLiteStablehloCompositeParams* params,
                                  int node_index) {
    QsaAttributes attr;
    if (!ReadAttributes(context, params, node_index, &attr)) {
      return kTfLiteError;
    }
    if (node->inputs->size != 5 || node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(context, "%s node #%d expects 5 inputs and 1 output",
                         kSparseSdpaName, node_index);
      return kTfLiteError;
    }
    if (!HasValidTensors(context, node) ||
        !ValidShapes(context, attr, node->inputs->data,
                     node->outputs->data[0])) {
      TF_LITE_KERNEL_LOG(context,
                         "%s node #%d: unsupported tensor types or shapes",
                         kSparseSdpaName, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  // Checks the types and shapes that Invoke() relies on. Called at delegation
  // time and again from Prepare() in case inputs were resized.
  static bool ValidShapes(TfLiteContext* context, const QsaAttributes& attr,
                          const int* inputs, int output_id) {
    for (int i = 0; i < 5; ++i) {
      if (!IsValidTensor(context, inputs[i])) {
        return false;
      }
    }
    const TfLiteTensor& q = context->tensors[inputs[0]];
    const TfLiteTensor& k = context->tensors[inputs[1]];
    const TfLiteTensor& v = context->tensors[inputs[2]];
    const TfLiteTensor& topk = context->tensors[inputs[3]];
    const TfLiteTensor& mask = context->tensors[inputs[4]];
    const TfLiteTensor& out = context->tensors[output_id];
    if (NumDims(q) != 4 || NumDims(k) != 4 || NumDims(v) != 4 ||
        NumDims(topk) != 3 || NumDims(mask) != 4) {
      return false;
    }
    if (!IsFloatType(q.type) || !IsFloatType(out.type) ||
        !IsFloatType(k.type) || k.type != v.type || topk.type != kTfLiteInt32) {
      return false;
    }
    const int batch = Dim(q, 0);
    const int num_q_heads = Dim(q, 1);
    const int seq_len = Dim(q, 2);
    const int head_dim = Dim(q, 3);
    const int num_kv_heads = Dim(k, 1);
    const int kv_len = Dim(k, attr.k_ts_idx);
    const int k_dim = Dim(k, attr.k_ts_idx == 2 ? 3 : 2);
    const int v_len = Dim(v, attr.v_ts_idx);
    const int v_dim = Dim(v, attr.v_ts_idx == 2 ? 3 : 2);
    return Dim(k, 0) == batch && Dim(v, 0) == batch &&
           Dim(v, 1) == num_kv_heads && num_q_heads % num_kv_heads == 0 &&
           k_dim == head_dim && v_dim == head_dim && v_len == kv_len &&
           kv_len >= attr.compress_ratio && Dim(topk, 0) == batch &&
           Dim(topk, 1) == seq_len &&
           MaskView::IsCompatible(mask, batch, seq_len, kv_len);
  }

  static std::unique_ptr<SparseSdpaKernel> Create(
      TfLiteContext* context, const TfLiteNode* node,
      const TfLiteStablehloCompositeParams* params, int node_index,
      pthreadpool_t threadpool) {
    QsaAttributes attr;
    if (IsSupported(context, node, params, node_index) != kTfLiteOk ||
        !ReadAttributes(context, params, node_index, &attr)) {
      return nullptr;
    }
    return std::unique_ptr<SparseSdpaKernel>(
        new SparseSdpaKernel(attr, node, threadpool));
  }

  TfLiteStatus Prepare(TfLiteContext* context) override {
    const int inputs[] = {query_id_, key_id_, value_id_, topk_id_, mask_id_};
    if (!ValidShapes(context, attr_, inputs, output_id_)) {
      TF_LITE_KERNEL_LOG(context, "%s: unsupported tensor shapes",
                         kSparseSdpaName);
      return kTfLiteError;
    }
    const TfLiteTensor& q = context->tensors[query_id_];
    if (!attr_.has_scale) {
      attr_.scale = 1.0f / std::sqrt(static_cast<float>(Dim(q, 3)));
    }
    TfLiteTensor& out = context->tensors[output_id_];
    TfLiteIntArray* shape = TfLiteIntArrayCreate(4);
    shape->data[0] = Dim(q, 0);
    shape->data[1] = Dim(q, 2);
    shape->data[2] = Dim(q, 1);
    shape->data[3] = Dim(q, 3);
    if (out.dims != nullptr && TfLiteIntArrayEqual(out.dims, shape)) {
      TfLiteIntArrayFree(shape);
      return kTfLiteOk;
    }
    return context->ResizeTensor(context, &out, shape);
  }

  TfLiteStatus Invoke(TfLiteContext* context) override {
    const TfLiteTensor& k = context->tensors[key_id_];
    if (k.type == kTfLiteFloat16) {
      return InvokeTyped<uint16_t>(context);
    }
    return InvokeTyped<float>(context);
  }

 private:
  SparseSdpaKernel(const QsaAttributes& attr, const TfLiteNode* node,
                   pthreadpool_t threadpool)
      : attr_(attr),
        query_id_(node->inputs->data[0]),
        key_id_(node->inputs->data[1]),
        value_id_(node->inputs->data[2]),
        topk_id_(node->inputs->data[3]),
        mask_id_(node->inputs->data[4]),
        output_id_(node->outputs->data[0]),
        threadpool_(threadpool) {}

  struct Geometry {
    int batch;
    int num_q_heads;
    int num_kv_heads;
    int groups;
    int seq_len;
    int head_dim;
    int kv_len;
    int ratio;
    int max_blocks;
    int block_tokens;
    int topk;
  };

  void InitScratch(const Geometry& g, SdpaScratch* s) const {
    s->selected.assign(g.max_blocks, 0);
    s->q.resize(static_cast<size_t>(g.groups) * g.head_dim);
    s->k.resize(static_cast<size_t>(kTokenTile) * g.head_dim);
    s->v.resize(std::max(g.groups, g.head_dim));
    s->max_logit.resize(g.groups);
    s->state.resize(static_cast<size_t>(g.groups) * (g.head_dim + 2));
  }

  // Builds the kept & visible KV token ranges of query row (b, s).
  void BuildSegments(const Geometry& g, const MaskView& mask,
                     const int32_t* topk, int b, int s,
                     SdpaScratch* scratch) const {
    const uint8_t* vis = mask.Row(b, s, &scratch->vis);
    uint8_t* selected = scratch->selected.data();
    const int32_t* row_topk =
        topk + (static_cast<size_t>(b) * g.seq_len + s) * g.topk;
    for (int i = 0; i < g.topk; ++i) {
      const int32_t idx = row_topk[i];
      if (idx >= 0 && idx < g.max_blocks) {
        selected[idx] = 1;
      }
    }
    std::vector<Segment>& segments = scratch->segments;
    segments.clear();
    auto append = [&segments](int32_t begin, int32_t end) {
      if (!segments.empty() && segments.back().end == begin) {
        segments.back().end = end;
      } else {
        segments.push_back({begin, end});
      }
    };
    const int ratio = g.ratio;
    for (int j = 0; j < g.max_blocks; ++j) {
      const int base = j * ratio;
      bool any = false;
      const bool complete = BlockVisibility(vis, base, ratio, &any);
      if (complete) {
        if (selected[j]) {
          append(base, base + ratio);
        }
      } else if (any) {
        // Incomplete blocks are always kept, restricted to visible tokens.
        for (int r = 0; r < ratio; ++r) {
          if (vis[base + r]) {
            append(base + r, base + r + 1);
          }
        }
      }
    }
    for (int t = g.block_tokens; t < g.kv_len; ++t) {
      if (vis[t]) {
        append(t, t + 1);
      }
    }
    for (int i = 0; i < g.topk; ++i) {
      const int32_t idx = row_topk[i];
      if (idx >= 0 && idx < g.max_blocks) {
        selected[idx] = 0;
      }
    }
  }

  // Loads the [G, D] query block of (b, s, kv_head) into `scratch->q`,
  // pre-multiplied by the attention scale.
  void LoadQuery(const Geometry& g, const TfLiteTensor& q, int b, int s,
                 int kv_head, SdpaScratch* scratch) const {
    float* dst = scratch->q.data();
    for (int gi = 0; gi < g.groups; ++gi) {
      const int h = kv_head * g.groups + gi;
      const size_t offset =
          ((static_cast<size_t>(b) * g.num_q_heads + h) * g.seq_len + s) *
          g.head_dim;
      if (q.type == kTfLiteFloat16) {
        ConvertToFloat(
            reinterpret_cast<const uint16_t*>(q.data.raw_const) + offset,
            dst + gi * g.head_dim, g.head_dim);
      } else {
        std::memcpy(dst + gi * g.head_dim,
                    reinterpret_cast<const float*>(q.data.raw_const) + offset,
                    g.head_dim * sizeof(float));
      }
      for (int d = 0; d < g.head_dim; ++d) {
        dst[gi * g.head_dim + d] *= attr_.scale;
      }
    }
  }

  static void ResetState(const Geometry& g, SoftmaxState* st) {
    std::fill(st->m, st->m + g.groups, -std::numeric_limits<float>::infinity());
    std::fill(st->l, st->l + g.groups, 0.0f);
    std::fill(st->acc, st->acc + static_cast<size_t>(g.groups) * g.head_dim,
              0.0f);
  }

  // Accumulates attention over `segments` into `st`. Logits of all tokens are
  // computed first; when the value cache is `[D, T]` (`v_ts_idx == 3`) values
  // are then gathered row by row, which keeps the strided reads TLB-friendly
  // and turns the weighted sum into `G` dot products per row.
  template <typename KV>
  void Attend(const Geometry& g, const KV* k_head, const KV* v_head,
              const Segment* segments, size_t num_segments,
              SdpaScratch* scratch, SoftmaxState* st) const {
    const int dim = g.head_dim;
    const int groups = g.groups;
    const size_t kv_len = g.kv_len;
    size_t num_tokens = 0;
    for (size_t si = 0; si < num_segments; ++si) {
      num_tokens += segments[si].end - segments[si].begin;
    }
    if (num_tokens == 0) {
      return;
    }
    // Softmax numerators, [G, num_tokens].
    scratch->logits.resize(num_tokens * groups);
    float* probs = scratch->logits.data();
    const float* q = scratch->q.data();
    float* kbuf = scratch->k.data();

    // 1. Logits; K is read in tiles of up to kTokenTile tokens.
    float* max_logit = scratch->max_logit.data();
    std::fill(max_logit, max_logit + groups,
              -std::numeric_limits<float>::infinity());
    size_t idx = 0;
    for (size_t si = 0; si < num_segments; ++si) {
      for (int t0 = segments[si].begin; t0 < segments[si].end;
           t0 += kTokenTile) {
        const int n = std::min(kTokenTile, segments[si].end - t0);
        if (attr_.k_ts_idx == 2) {
          ConvertToFloat(k_head + static_cast<size_t>(t0) * dim, kbuf, n * dim);
        } else {
          for (int d = 0; d < dim; ++d) {
            const KV* src = k_head + d * kv_len + t0;
            for (int j = 0; j < n; ++j) {
              kbuf[j * dim + d] = ToFloat(src[j]);
            }
          }
        }
        for (int gi = 0; gi < groups; ++gi) {
          float* pg = probs + gi * num_tokens + idx;
          for (int j = 0; j < n; ++j) {
            pg[j] = Dot(q + gi * dim, kbuf + j * dim, dim);
            max_logit[gi] = std::max(max_logit[gi], pg[j]);
          }
        }
        idx += n;
      }
    }

    // 2. Online-softmax update.
    for (int gi = 0; gi < groups; ++gi) {
      if (max_logit[gi] > st->m[gi]) {
        if (st->l[gi] != 0.0f) {
          const float corr = std::exp(st->m[gi] - max_logit[gi]);
          st->l[gi] *= corr;
          float* acc = st->acc + gi * dim;
          for (int d = 0; d < dim; ++d) {
            acc[d] *= corr;
          }
        }
        st->m[gi] = max_logit[gi];
      }
      float* pg = probs + gi * num_tokens;
      const float m = st->m[gi];
      float sum = 0.0f;
      for (size_t i = 0; i < num_tokens; ++i) {
        pg[i] = std::exp(pg[i] - m);
        sum += pg[i];
      }
      st->l[gi] += sum;
    }

    // 3. Weighted sum of values.
    if (attr_.v_ts_idx == 3) {
      scratch->v_raw.resize(num_tokens * sizeof(KV));
      scratch->v_row.resize(num_tokens);
      KV* raw = reinterpret_cast<KV*>(scratch->v_raw.data());
      float* vrow = scratch->v_row.data();
      for (int d = 0; d < dim; ++d) {
        const KV* row = v_head + d * kv_len;
        size_t i = 0;
        for (size_t si = 0; si < num_segments; ++si) {
          const int len = segments[si].end - segments[si].begin;
          std::memcpy(raw + i, row + segments[si].begin, len * sizeof(KV));
          i += len;
        }
        ConvertToFloat(raw, vrow, static_cast<int>(num_tokens));
        for (int gi = 0; gi < groups; ++gi) {
          st->acc[gi * dim + d] +=
              Dot(probs + gi * num_tokens, vrow, static_cast<int>(num_tokens));
        }
      }
    } else {
      float* vbuf = scratch->v.data();
      size_t i = 0;
      for (size_t si = 0; si < num_segments; ++si) {
        for (int t = segments[si].begin; t < segments[si].end; ++t, ++i) {
          ConvertToFloat(v_head + static_cast<size_t>(t) * dim, vbuf, dim);
          for (int gi = 0; gi < groups; ++gi) {
            const float p = probs[gi * num_tokens + i];
            float* acc = st->acc + gi * dim;
            for (int d = 0; d < dim; ++d) {
              acc[d] += p * vbuf[d];
            }
          }
        }
      }
    }
  }

  void WriteOutput(const Geometry& g, TfLiteTensor& out, int b, int s,
                   int kv_head, const SoftmaxState& st) const {
    for (int gi = 0; gi < g.groups; ++gi) {
      const int h = kv_head * g.groups + gi;
      const size_t offset =
          ((static_cast<size_t>(b) * g.seq_len + s) * g.num_q_heads + h) *
          g.head_dim;
      const float inv_l = st.l[gi] > 0.0f ? 1.0f / st.l[gi] : 0.0f;
      const float* acc = st.acc + gi * g.head_dim;
      if (out.type == kTfLiteFloat16) {
        uint16_t* dst = reinterpret_cast<uint16_t*>(out.data.raw) + offset;
        for (int d = 0; d < g.head_dim; ++d) {
          dst[d] = fp16_ieee_from_fp32_value(acc[d] * inv_l);
        }
      } else {
        float* dst = reinterpret_cast<float*>(out.data.raw) + offset;
        for (int d = 0; d < g.head_dim; ++d) {
          dst[d] = acc[d] * inv_l;
        }
      }
    }
  }

  SoftmaxState StateView(const Geometry& g, float* base) const {
    return SoftmaxState{base, base + g.groups, base + 2 * g.groups};
  }

  template <typename KV>
  TfLiteStatus InvokeTyped(TfLiteContext* context) {
    const TfLiteTensor& q = context->tensors[query_id_];
    const TfLiteTensor& k = context->tensors[key_id_];
    const TfLiteTensor& v = context->tensors[value_id_];
    const TfLiteTensor& topk_tensor = context->tensors[topk_id_];
    const TfLiteTensor& mask_tensor = context->tensors[mask_id_];
    TfLiteTensor& out = context->tensors[output_id_];
    if (q.data.raw_const == nullptr || k.data.raw_const == nullptr ||
        v.data.raw_const == nullptr || topk_tensor.data.raw_const == nullptr ||
        mask_tensor.data.raw_const == nullptr || out.data.raw == nullptr) {
      TF_LITE_KERNEL_LOG(context, "%s received a null tensor data pointer",
                         kSparseSdpaName);
      return kTfLiteError;
    }

    Geometry g;
    g.batch = Dim(q, 0);
    g.num_q_heads = Dim(q, 1);
    g.seq_len = Dim(q, 2);
    g.head_dim = Dim(q, 3);
    g.num_kv_heads = Dim(k, 1);
    g.groups = g.num_q_heads / g.num_kv_heads;
    g.kv_len = Dim(k, attr_.k_ts_idx);
    g.ratio = attr_.compress_ratio;
    g.max_blocks = g.kv_len / g.ratio;
    g.block_tokens = g.max_blocks * g.ratio;
    g.topk = Dim(topk_tensor, 2);
    if (NumElements(out) != NumElements(q)) {
      TF_LITE_KERNEL_LOG(context, "%s output size mismatch", kSparseSdpaName);
      return kTfLiteError;
    }

    const KV* k_data = reinterpret_cast<const KV*>(k.data.raw_const);
    const KV* v_data = reinterpret_cast<const KV*>(v.data.raw_const);
    const int32_t* topk = topk_tensor.data.i32;
    const MaskView mask(mask_tensor);
    const size_t head_stride = static_cast<size_t>(g.kv_len) * g.head_dim;
    auto head_ptr = [&](const KV* base, int b, int h) {
      return base + (static_cast<size_t>(b) * g.num_kv_heads + h) * head_stride;
    };

    const size_t num_threads = NumThreads(threadpool_);
    const size_t num_rows = static_cast<size_t>(g.batch) * g.seq_len;
    // Scratch buffers persist across Invoke() calls and only grow.
    if (scratch_.size() < num_threads) {
      scratch_.resize(num_threads);
    }
    std::vector<SdpaScratch>& scratch = scratch_;
    for (size_t t = 0; t < num_threads; ++t) {
      InitScratch(g, &scratch[t]);
    }

    if (num_rows * g.num_kv_heads >= 2 * num_threads) {
      // Row mode: one work item per query row; all KV heads of the row reuse
      // the same token ranges.
      ParallelForWithThread(
          threadpool_, num_rows, [&](size_t thread, size_t row) {
            SdpaScratch& s = scratch[thread];
            const int b = static_cast<int>(row / g.seq_len);
            const int sq = static_cast<int>(row % g.seq_len);
            BuildSegments(g, mask, topk, b, sq, &s);
            SoftmaxState st = StateView(g, s.state.data());
            for (int h = 0; h < g.num_kv_heads; ++h) {
              LoadQuery(g, q, b, sq, h, &s);
              ResetState(g, &st);
              Attend(g, head_ptr(k_data, b, h), head_ptr(v_data, b, h),
                     s.segments.data(), s.segments.size(), &s, &st);
              WriteOutput(g, out, b, sq, h, st);
            }
          });
      return kTfLiteOk;
    }

    // Split mode (decode): split the kept tokens of each row into chunks that
    // are processed in parallel and merged afterwards.
    const size_t state_size = static_cast<size_t>(g.groups) * (g.head_dim + 2);
    std::vector<Segment>& chunk_segments = chunk_segments_;
    std::vector<size_t>& chunk_offsets = chunk_offsets_;
    std::vector<float>& partials = partials_;
    for (size_t row = 0; row < num_rows; ++row) {
      const int b = static_cast<int>(row / g.seq_len);
      const int sq = static_cast<int>(row % g.seq_len);
      BuildSegments(g, mask, topk, b, sq, &scratch[0]);
      const std::vector<Segment>& segments = scratch[0].segments;
      size_t total = 0;
      for (const Segment& seg : segments) {
        total += seg.end - seg.begin;
      }
      const size_t wanted =
          (2 * num_threads + g.num_kv_heads - 1) / g.num_kv_heads;
      const size_t num_chunks =
          std::max<size_t>(1, std::min(wanted, total / kMinTokensPerChunk));
      const size_t chunk_tokens = (total + num_chunks - 1) / num_chunks;
      chunk_segments.clear();
      chunk_offsets.assign(1, 0);
      size_t in_chunk = 0;
      for (const Segment& seg : segments) {
        int32_t begin = seg.begin;
        while (begin < seg.end) {
          const int32_t end = static_cast<int32_t>(
              std::min<size_t>(seg.end, begin + (chunk_tokens - in_chunk)));
          chunk_segments.push_back({begin, end});
          in_chunk += end - begin;
          begin = end;
          if (in_chunk == chunk_tokens) {
            chunk_offsets.push_back(chunk_segments.size());
            in_chunk = 0;
          }
        }
      }
      if (chunk_offsets.back() != chunk_segments.size()) {
        chunk_offsets.push_back(chunk_segments.size());
      }
      const size_t chunks = std::max<size_t>(1, chunk_offsets.size() - 1);
      if (chunk_offsets.size() == 1) {
        chunk_offsets.push_back(0);
      }
      partials.resize(chunks * g.num_kv_heads * state_size);
      ParallelForWithThread(
          threadpool_, chunks * g.num_kv_heads,
          [&](size_t thread, size_t item) {
            SdpaScratch& s = scratch[thread];
            const int h = static_cast<int>(item / chunks);
            const size_t c = item % chunks;
            LoadQuery(g, q, b, sq, h, &s);
            SoftmaxState st = StateView(g, partials.data() + item * state_size);
            ResetState(g, &st);
            Attend(g, head_ptr(k_data, b, h), head_ptr(v_data, b, h),
                   chunk_segments.data() + chunk_offsets[c],
                   chunk_offsets[c + 1] - chunk_offsets[c], &s, &st);
          });
      // Merge the partial softmax states of each KV head.
      SoftmaxState merged = StateView(g, scratch[0].state.data());
      for (int h = 0; h < g.num_kv_heads; ++h) {
        ResetState(g, &merged);
        for (size_t c = 0; c < chunks; ++c) {
          const SoftmaxState part =
              StateView(g, partials.data() + (h * chunks + c) * state_size);
          for (int gi = 0; gi < g.groups; ++gi) {
            if (part.l[gi] == 0.0f) {
              continue;
            }
            const float m = std::max(merged.m[gi], part.m[gi]);
            const float a =
                merged.l[gi] == 0.0f ? 0.0f : std::exp(merged.m[gi] - m);
            const float p = std::exp(part.m[gi] - m);
            float* acc = merged.acc + gi * g.head_dim;
            const float* pacc = part.acc + gi * g.head_dim;
            for (int d = 0; d < g.head_dim; ++d) {
              acc[d] = acc[d] * a + pacc[d] * p;
            }
            merged.l[gi] = merged.l[gi] * a + part.l[gi] * p;
            merged.m[gi] = m;
          }
        }
        WriteOutput(g, out, b, sq, h, merged);
      }
    }
    return kTfLiteOk;
  }

  QsaAttributes attr_;
  int query_id_;
  int key_id_;
  int value_id_;
  int topk_id_;
  int mask_id_;
  int output_id_;
  pthreadpool_t threadpool_;
  // Per-thread and split-mode scratch, reused across Invoke() calls.
  std::vector<SdpaScratch> scratch_;
  std::vector<Segment> chunk_segments_;
  std::vector<size_t> chunk_offsets_;
  std::vector<float> partials_;
};

// ---------------------------------------------------------------------------
// odml.qsa_indexer_update
// ---------------------------------------------------------------------------

struct IndexerScratch {
  std::vector<uint8_t> vis;
  std::vector<float> q;     // [H, D].
  std::vector<float> pool;  // [D].
  std::vector<std::pair<float, int32_t>> candidates;
};

class IndexerUpdateKernel : public QsaDelegateKernel::Impl {
 public:
  enum Input {
    kQIdx = 0,
    kKIdxNew,
    kRingK,
    kBlockKPool,
    kKNormWeight,
    kBlockCos,
    kBlockSin,
    kStartPos,
    kAttentionMask,
    kNumInputs,
  };
  enum Output {
    kTopk = 0,
    kNewRingK,
    kNewBlockKPool,
    kNumOutputs,
  };

  static TfLiteStatus IsSupported(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLiteStablehloCompositeParams* params,
                                  int node_index) {
    QsaAttributes attr;
    if (!ReadAttributes(context, params, node_index, &attr)) {
      return kTfLiteError;
    }
    if (node->inputs->size != kNumInputs ||
        node->outputs->size != kNumOutputs) {
      TF_LITE_KERNEL_LOG(context, "%s node #%d expects %d inputs, %d outputs",
                         kIndexerUpdateName, node_index, kNumInputs,
                         kNumOutputs);
      return kTfLiteError;
    }
    if (!HasValidTensors(context, node) ||
        !ValidShapes(context, attr, node->inputs->data, node->outputs->data)) {
      TF_LITE_KERNEL_LOG(context,
                         "%s node #%d: unsupported tensor types or shapes",
                         kIndexerUpdateName, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  // Checks the types and shapes that Invoke() relies on. Called at delegation
  // time and again from Prepare() in case inputs were resized.
  static bool ValidShapes(TfLiteContext* context, const QsaAttributes& attr,
                          const int* inputs, const int* outputs) {
    for (int i = 0; i < kNumInputs; ++i) {
      if (!IsValidTensor(context, inputs[i])) {
        return false;
      }
    }
    auto in = [&](int i) -> const TfLiteTensor& {
      return context->tensors[inputs[i]];
    };
    auto out = [&](int i) -> const TfLiteTensor& {
      return context->tensors[outputs[i]];
    };
    const TfLiteTensor& q = in(kQIdx);
    const TfLiteTensor& k_new = in(kKIdxNew);
    const TfLiteTensor& ring = in(kRingK);
    const TfLiteTensor& pool = in(kBlockKPool);
    const TfLiteTensor& weight = in(kKNormWeight);
    const TfLiteTensor& cos = in(kBlockCos);
    const TfLiteTensor& sin = in(kBlockSin);
    const TfLiteTensor& start_pos = in(kStartPos);
    const TfLiteTensor& mask = in(kAttentionMask);
    if (NumDims(q) != 4 || NumDims(k_new) != 3 || NumDims(ring) != 3 ||
        NumDims(pool) != 3 || NumDims(weight) < 1 || NumDims(cos) != 3 ||
        NumDims(sin) != 3 || NumDims(mask) != 4) {
      return false;
    }
    if (!IsFloatType(q.type) || !IsFloatType(ring.type) ||
        k_new.type != ring.type || pool.type != ring.type ||
        !IsFloatType(weight.type) || !IsFloatType(cos.type) ||
        sin.type != cos.type || start_pos.type != kTfLiteInt32 ||
        out(kTopk).type != kTfLiteInt32 || out(kNewRingK).type != ring.type ||
        out(kNewBlockKPool).type != pool.type) {
      return false;
    }
    const int batch = Dim(q, 0);
    const int seq_len = Dim(q, 1);
    const int dim = Dim(q, 3);
    const int max_blocks = Dim(pool, 1);
    const int rotary_dim = Dim(cos, 2);
    const int kv_len = Dim(mask, 3);
    return Dim(k_new, 0) == batch && Dim(k_new, 1) == seq_len &&
           Dim(k_new, 2) == dim && Dim(ring, 0) == batch &&
           Dim(ring, 1) == attr.compress_ratio && Dim(ring, 2) == dim &&
           Dim(pool, 0) == batch && Dim(pool, 2) == dim &&
           NumElements(weight) == static_cast<size_t>(dim) &&
           Dim(cos, 0) == 1 && Dim(cos, 1) == max_blocks && rotary_dim <= dim &&
           rotary_dim % 2 == 0 && TfLiteIntArrayEqual(cos.dims, sin.dims) &&
           NumElements(start_pos) == 1 &&
           static_cast<int64_t>(max_blocks) * attr.compress_ratio <= kv_len &&
           MaskView::IsCompatible(mask, batch, seq_len, kv_len);
  }

  static std::unique_ptr<IndexerUpdateKernel> Create(
      TfLiteContext* context, const TfLiteNode* node,
      const TfLiteStablehloCompositeParams* params, int node_index,
      pthreadpool_t threadpool) {
    QsaAttributes attr;
    if (IsSupported(context, node, params, node_index) != kTfLiteOk ||
        !ReadAttributes(context, params, node_index, &attr)) {
      return nullptr;
    }
    return std::unique_ptr<IndexerUpdateKernel>(
        new IndexerUpdateKernel(attr, node, threadpool));
  }

  TfLiteStatus Prepare(TfLiteContext* context) override {
    if (!ValidShapes(context, attr_, inputs_, outputs_)) {
      TF_LITE_KERNEL_LOG(context, "%s: unsupported tensor shapes",
                         kIndexerUpdateName);
      return kTfLiteError;
    }
    const TfLiteTensor& q = context->tensors[inputs_[kQIdx]];
    scale_ = attr_.has_scale ? attr_.scale
                             : 1.0f / std::sqrt(static_cast<float>(Dim(q, 3)));
    TfLiteIntArray* topk_shape = TfLiteIntArrayCreate(3);
    topk_shape->data[0] = Dim(q, 0);
    topk_shape->data[1] = Dim(q, 1);
    topk_shape->data[2] = attr_.block_topk;
    TF_LITE_ENSURE_STATUS(ResizeIfNeeded(context, outputs_[kTopk], topk_shape));
    TF_LITE_ENSURE_STATUS(ResizeIfNeeded(
        context, outputs_[kNewRingK],
        TfLiteIntArrayCopy(context->tensors[inputs_[kRingK]].dims)));
    return ResizeIfNeeded(
        context, outputs_[kNewBlockKPool],
        TfLiteIntArrayCopy(context->tensors[inputs_[kBlockKPool]].dims));
  }

  TfLiteStatus Invoke(TfLiteContext* context) override {
    for (int i = 0; i < kNumInputs; ++i) {
      if (context->tensors[inputs_[i]].data.raw_const == nullptr) {
        TF_LITE_KERNEL_LOG(context, "%s input #%d has no data",
                           kIndexerUpdateName, i);
        return kTfLiteError;
      }
    }
    for (int i = 0; i < kNumOutputs; ++i) {
      if (context->tensors[outputs_[i]].data.raw == nullptr) {
        TF_LITE_KERNEL_LOG(context, "%s output #%d has no data",
                           kIndexerUpdateName, i);
        return kTfLiteError;
      }
    }
    if (context->tensors[inputs_[kRingK]].type == kTfLiteFloat16) {
      return InvokeTyped<uint16_t>(context);
    }
    return InvokeTyped<float>(context);
  }

 private:
  IndexerUpdateKernel(const QsaAttributes& attr, const TfLiteNode* node,
                      pthreadpool_t threadpool)
      : attr_(attr), threadpool_(threadpool) {
    for (int i = 0; i < kNumInputs; ++i) {
      inputs_[i] = node->inputs->data[i];
    }
    for (int i = 0; i < kNumOutputs; ++i) {
      outputs_[i] = node->outputs->data[i];
    }
  }

  static TfLiteStatus ResizeIfNeeded(TfLiteContext* context, int tensor_id,
                                     TfLiteIntArray* shape) {
    TfLiteTensor& t = context->tensors[tensor_id];
    if (t.dims != nullptr && TfLiteIntArrayEqual(t.dims, shape)) {
      TfLiteIntArrayFree(shape);
      return kTfLiteOk;
    }
    return context->ResizeTensor(context, &t, shape);
  }

  // Mean-pools `ratio` raw keys, applies RMSNorm and partial RoPE, rounding to
  // the activation type exactly where the reference decomposition does.
  // `raw` is [ratio, D] in fp32 (exactly representable in `act_type`).
  void PoolBlock(const float* raw, const TfLiteTensor& weight,
                 const TfLiteTensor& cos, const TfLiteTensor& sin, int block,
                 TfLiteType act_type, int dim, float* out) const {
    const int ratio = attr_.compress_ratio;
    float sum_sq = 0.0f;
    for (int d = 0; d < dim; ++d) {
      float sum = 0.0f;
      for (int r = 0; r < ratio; ++r) {
        sum += raw[r * dim + d];
      }
      out[d] = RoundTo(act_type, sum / static_cast<float>(ratio));
      sum_sq += out[d] * out[d];
    }
    const float inv_rms =
        1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + attr_.rms_norm_eps);
    for (int d = 0; d < dim; ++d) {
      const float normed = RoundTo(act_type, out[d] * inv_rms);
      out[d] =
          RoundTo(act_type, normed * RoundTo(act_type, ReadFloat(weight, d)));
    }
    const int rotary_dim = Dim(cos, 2);
    const int half = rotary_dim / 2;
    const size_t base = static_cast<size_t>(block) * rotary_dim;
    float rotated[2];
    for (int i = 0; i < half; ++i) {
      const float x1 = out[i];
      const float x2 = out[i + half];
      rotated[0] =
          x1 * ReadFloat(cos, base + i) - x2 * ReadFloat(sin, base + i);
      rotated[1] = x2 * ReadFloat(cos, base + i + half) +
                   x1 * ReadFloat(sin, base + i + half);
      out[i] = RoundTo(act_type, rotated[0]);
      out[i + half] = RoundTo(act_type, rotated[1]);
    }
  }

  // Scores the complete blocks of one query row and writes the sorted top-k
  // block indices (padded with -1).
  template <typename T>
  void ScoreRow(const TfLiteTensor& q, const T* pool, const uint8_t* vis, int b,
                int s, int seq_len, int num_heads, int dim, int max_blocks,
                int32_t* out_row, IndexerScratch* scratch) const {
    const int ratio = attr_.compress_ratio;
    std::vector<std::pair<float, int32_t>>& cand = scratch->candidates;
    cand.clear();
    for (int j = 0; j < max_blocks; ++j) {
      bool any = false;
      if (BlockVisibility(vis, j * ratio, ratio, &any)) {
        cand.push_back({0.0f, j});
      }
    }
    LoadQuery(q, b, s, seq_len, num_heads, dim, scratch);
    ScoreCandidates(pool, num_heads, dim, cand.data(), cand.size(), scratch);
    SelectTopk(&cand, max_blocks, out_row);
  }

  void LoadQuery(const TfLiteTensor& q, int b, int s, int seq_len,
                 int num_heads, int dim, IndexerScratch* scratch) const {
    const size_t n = static_cast<size_t>(num_heads) * dim;
    scratch->q.resize(n);
    const size_t offset = (static_cast<size_t>(b) * seq_len + s) * n;
    for (size_t i = 0; i < n; ++i) {
      scratch->q[i] = ReadFloat(q, offset + i);
    }
  }

  template <typename T>
  void ScoreCandidates(const T* pool, int num_heads, int dim,
                       std::pair<float, int32_t>* cand, size_t count,
                       IndexerScratch* scratch) const {
    scratch->pool.resize(dim);
    float* p = scratch->pool.data();
    const float* q = scratch->q.data();
    for (size_t c = 0; c < count; ++c) {
      ConvertToFloat(pool + static_cast<size_t>(cand[c].second) * dim, p, dim);
      float score = 0.0f;
      for (int h = 0; h < num_heads; ++h) {
        score += std::max(Dot(q + h * dim, p, dim), 0.0f);
      }
      // NaN scores would break the strict weak ordering used by top-k.
      const float scaled = score * scale_;
      cand[c].first =
          std::isnan(scaled) ? -std::numeric_limits<float>::infinity() : scaled;
    }
  }

  void SelectTopk(std::vector<std::pair<float, int32_t>>* cand, int max_blocks,
                  int32_t* out_row) const {
    // Highest score first; ties prefer the lower block index.
    auto better = [](const std::pair<float, int32_t>& a,
                     const std::pair<float, int32_t>& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    const size_t k_blocks =
        std::min<size_t>(attr_.block_topk, static_cast<size_t>(max_blocks));
    const size_t keep = std::min(k_blocks, cand->size());
    if (keep < cand->size()) {
      std::nth_element(cand->begin(), cand->begin() + keep, cand->end(),
                       better);
    }
    std::sort(cand->begin(), cand->begin() + keep, better);
    for (size_t i = 0; i < keep; ++i) {
      out_row[i] = (*cand)[i].second;
    }
    std::fill(out_row + keep, out_row + attr_.block_topk, -1);
  }

  template <typename T>
  TfLiteStatus InvokeTyped(TfLiteContext* context) {
    const TfLiteTensor& q = context->tensors[inputs_[kQIdx]];
    const TfLiteTensor& k_new_tensor = context->tensors[inputs_[kKIdxNew]];
    const TfLiteTensor& ring_tensor = context->tensors[inputs_[kRingK]];
    const TfLiteTensor& pool_tensor = context->tensors[inputs_[kBlockKPool]];
    const TfLiteTensor& weight = context->tensors[inputs_[kKNormWeight]];
    const TfLiteTensor& cos = context->tensors[inputs_[kBlockCos]];
    const TfLiteTensor& sin = context->tensors[inputs_[kBlockSin]];
    const TfLiteTensor& start_pos = context->tensors[inputs_[kStartPos]];
    const TfLiteTensor& mask_tensor = context->tensors[inputs_[kAttentionMask]];
    TfLiteTensor& topk_tensor = context->tensors[outputs_[kTopk]];
    TfLiteTensor& new_ring_tensor = context->tensors[outputs_[kNewRingK]];
    TfLiteTensor& new_pool_tensor = context->tensors[outputs_[kNewBlockKPool]];

    const int batch = Dim(q, 0);
    const int seq_len = Dim(q, 1);
    const int num_heads = Dim(q, 2);
    const int dim = Dim(q, 3);
    const int ratio = attr_.compress_ratio;
    const int max_blocks = Dim(pool_tensor, 1);
    const int kv_len = Dim(mask_tensor, 3);
    const TfLiteType act_type = ring_tensor.type;
    const int32_t pos0 = start_pos.data.i32[0];
    const MaskView mask(mask_tensor);

    const T* k_new = reinterpret_cast<const T*>(k_new_tensor.data.raw_const);
    const T* ring = reinterpret_cast<const T*>(ring_tensor.data.raw_const);
    const T* pool = reinterpret_cast<const T*>(pool_tensor.data.raw_const);
    T* new_ring = reinterpret_cast<T*>(new_ring_tensor.data.raw);
    T* new_pool = reinterpret_cast<T*>(new_pool_tensor.data.raw);
    const size_t ring_size = static_cast<size_t>(ratio) * dim;
    const size_t pool_size = static_cast<size_t>(max_blocks) * dim;

    auto floor_div = [](int64_t a, int64_t b) {
      return a >= 0 ? a / b : -((-a + b - 1) / b);
    };
    // The reference decomposition reads the diagonal of batch entry 0.
    auto token_visible = [&](int row, int64_t t) {
      return t >= 0 && t < kv_len && mask.At(0, row, static_cast<int>(t));
    };

    // 1. Compute the new ring buffer and the blocks completed by this step
    //    from the inputs only (outputs may alias the inputs).
    std::vector<float>& raw = raw_;
    std::vector<float>& ring_out = ring_out_;
    std::vector<int32_t>& written_blocks = written_blocks_;
    std::vector<float>& written_values = written_values_;
    raw.resize(ring_size);
    ring_out.resize(ring_size);
    written_blocks.clear();
    written_values.clear();
    for (int b = 0; b < batch; ++b) {
      const T* ring_b = ring + b * ring_size;
      const T* k_new_b = k_new + static_cast<size_t>(b) * seq_len * dim;
      float* ring_out_b = ring_out.data();
      if (seq_len == 1) {
        const bool tok_vis = token_visible(0, pos0);
        const int slot =
            static_cast<int>(pos0 - floor_div(pos0, ratio) * ratio);
        for (int r = 0; r < ratio; ++r) {
          const bool take_new = tok_vis && r == slot;
          for (int d = 0; d < dim; ++d) {
            ring_out_b[r * dim + d] =
                take_new ? ToFloat(k_new_b[d]) : ToFloat(ring_b[r * dim + d]);
          }
        }
        const int64_t block = floor_div(pos0, ratio);
        const bool block_end =
            tok_vis &&
            floor_div(static_cast<int64_t>(pos0) + 1, ratio) * ratio ==
                static_cast<int64_t>(pos0) + 1;
        if (block_end && block >= 0 && block < max_blocks) {
          written_blocks.push_back(static_cast<int32_t>(block));
          written_values.resize(written_blocks.size() * dim);
          PoolBlock(ring_out_b, weight, cos, sin, static_cast<int>(block),
                    act_type, dim,
                    written_values.data() + (written_blocks.size() - 1) * dim);
        }
      } else {
        // Prefill: the valid length is derived from the causal diagonal.
        int valid_len = 0;
        std::vector<uint8_t>& tok_valid = tok_valid_;
        tok_valid.assign(seq_len, 0);
        for (int i = 0; i < seq_len; ++i) {
          tok_valid[i] = token_visible(i, static_cast<int64_t>(pos0) + i);
          if (tok_valid[i]) {
            valid_len = i + 1;
          }
        }
        const int64_t start_block = floor_div(pos0, ratio);
        const int64_t start_slot = pos0 - start_block * ratio;
        auto raw_slot = [&](int64_t t, int d) -> float {
          const int64_t i = t - pos0;
          if (i >= 0 && i < seq_len && tok_valid[i]) {
            return ToFloat(k_new_b[i * dim + d]);
          }
          if (floor_div(t, ratio) == start_block) {
            const int64_t r = t - start_block * ratio;
            if (r < start_slot) {
              return ToFloat(ring_b[r * dim + d]);
            }
          }
          return 0.0f;
        };
        const int64_t end_pos = static_cast<int64_t>(pos0) + valid_len;
        const int64_t num_complete = floor_div(end_pos, ratio);
        for (int64_t blk = std::max<int64_t>(start_block, 0);
             blk < std::min<int64_t>(num_complete, max_blocks); ++blk) {
          for (int r = 0; r < ratio; ++r) {
            for (int d = 0; d < dim; ++d) {
              raw[r * dim + d] = raw_slot(blk * ratio + r, d);
            }
          }
          written_blocks.push_back(static_cast<int32_t>(blk));
          written_values.resize(written_blocks.size() * dim);
          PoolBlock(raw.data(), weight, cos, sin, static_cast<int>(blk),
                    act_type, dim,
                    written_values.data() + (written_blocks.size() - 1) * dim);
        }
        const int64_t tail_block = floor_div(end_pos, ratio);
        for (int r = 0; r < ratio; ++r) {
          for (int d = 0; d < dim; ++d) {
            ring_out_b[r * dim + d] = tail_block >= 0 && tail_block < max_blocks
                                          ? raw_slot(tail_block * ratio + r, d)
                                          : 0.0f;
          }
        }
      }
      // 2. Write the outputs for this batch entry.
      T* new_ring_b = new_ring + b * ring_size;
      for (size_t i = 0; i < ring_size; ++i) {
        new_ring_b[i] = FromFloat<T>(ring_out_b[i]);
      }
      T* new_pool_b = new_pool + b * pool_size;
      if (new_pool_b != pool + b * pool_size) {
        std::memcpy(new_pool_b, pool + b * pool_size, pool_size * sizeof(T));
      }
      for (size_t w = 0; w < written_blocks.size(); ++w) {
        T* dst = new_pool_b + static_cast<size_t>(written_blocks[w]) * dim;
        for (int d = 0; d < dim; ++d) {
          dst[d] = FromFloat<T>(written_values[w * dim + d]);
        }
      }
      written_blocks.clear();
      written_values.clear();
    }

    // 3. Score complete blocks against the updated pool and select top-k.
    const size_t num_threads = NumThreads(threadpool_);
    const size_t num_rows = static_cast<size_t>(batch) * seq_len;
    if (scratch_.size() < num_threads) {
      scratch_.resize(num_threads);
    }
    std::vector<IndexerScratch>& scratch = scratch_;
    int32_t* topk = topk_tensor.data.i32;
    if (num_rows > 1) {
      ParallelForWithThread(threadpool_, num_rows,
                            [&](size_t thread, size_t row) {
                              IndexerScratch& s = scratch[thread];
                              const int b = static_cast<int>(row / seq_len);
                              const int sq = static_cast<int>(row % seq_len);
                              const uint8_t* vis = mask.Row(b, sq, &s.vis);
                              ScoreRow(q, new_pool + b * pool_size, vis, b, sq,
                                       seq_len, num_heads, dim, max_blocks,
                                       topk + row * attr_.block_topk, &s);
                            });
      return kTfLiteOk;
    }

    // Single row (decode): score the complete blocks in parallel chunks.
    IndexerScratch& s0 = scratch[0];
    const uint8_t* vis = mask.Row(0, 0, &s0.vis);
    std::vector<std::pair<float, int32_t>>& cand = candidates_;
    cand.clear();
    for (int j = 0; j < max_blocks; ++j) {
      bool any = false;
      if (BlockVisibility(vis, j * ratio, ratio, &any)) {
        cand.push_back({0.0f, j});
      }
    }
    const size_t num_chunks = std::max<size_t>(
        1, std::min(2 * num_threads, cand.size() / kMinBlocksPerChunk));
    const size_t per_chunk = (cand.size() + num_chunks - 1) / num_chunks;
    ParallelForWithThread(
        threadpool_, num_chunks, [&](size_t thread, size_t c) {
          IndexerScratch& s = scratch[thread];
          const size_t begin = c * per_chunk;
          const size_t end = std::min(cand.size(), begin + per_chunk);
          if (begin >= end) {
            return;
          }
          LoadQuery(q, 0, 0, seq_len, num_heads, dim, &s);
          ScoreCandidates(new_pool, num_heads, dim, cand.data() + begin,
                          end - begin, &s);
        });
    SelectTopk(&cand, max_blocks, topk);
    return kTfLiteOk;
  }

  QsaAttributes attr_;
  float scale_ = 1.0f;
  int inputs_[kNumInputs];
  int outputs_[kNumOutputs];
  pthreadpool_t threadpool_;
  // Scratch buffers reused across Invoke() calls.
  std::vector<IndexerScratch> scratch_;
  std::vector<float> raw_;
  std::vector<float> ring_out_;
  std::vector<int32_t> written_blocks_;
  std::vector<float> written_values_;
  std::vector<uint8_t> tok_valid_;
  std::vector<std::pair<float, int32_t>> candidates_;
};

}  // namespace

QsaDelegateKernel::QsaDelegateKernel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

QsaDelegateKernel::~QsaDelegateKernel() = default;

bool QsaDelegateKernel::IsQsaNode(const TfLiteRegistration* registration,
                                  const TfLiteNode* node) {
  const TfLiteStablehloCompositeParams* params =
      GetCompositeParams(registration, node);
  return params != nullptr &&
         (std::strcmp(params->name, kSparseSdpaName) == 0 ||
          std::strcmp(params->name, kIndexerUpdateName) == 0);
}

TfLiteStatus QsaDelegateKernel::IsSupported(
    TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration, int node_index) {
  const TfLiteStablehloCompositeParams* params =
      GetCompositeParams(registration, node);
  if (params == nullptr || node->inputs == nullptr ||
      node->outputs == nullptr) {
    return kTfLiteError;
  }
  if (std::strcmp(params->name, kSparseSdpaName) == 0) {
    return SparseSdpaKernel::IsSupported(context, node, params, node_index);
  }
  if (std::strcmp(params->name, kIndexerUpdateName) == 0) {
    return IndexerUpdateKernel::IsSupported(context, node, params, node_index);
  }
  return kTfLiteError;
}

std::unique_ptr<QsaDelegateKernel> QsaDelegateKernel::Create(
    TfLiteContext* context, const TfLiteDelegateParams* params,
    pthreadpool_t threadpool) {
  if (params->nodes_to_replace == nullptr ||
      params->nodes_to_replace->size != 1) {
    return nullptr;
  }
  const int node_index = params->nodes_to_replace->data[0];
  TfLiteNode* node = nullptr;
  TfLiteRegistration* registration = nullptr;
  if (context->GetNodeAndRegistration(context, node_index, &node,
                                      &registration) != kTfLiteOk) {
    return nullptr;
  }
  const TfLiteStablehloCompositeParams* composite_params =
      GetCompositeParams(registration, node);
  if (composite_params == nullptr || node->inputs == nullptr ||
      node->outputs == nullptr) {
    return nullptr;
  }
  std::unique_ptr<Impl> impl;
  if (std::strcmp(composite_params->name, kSparseSdpaName) == 0) {
    impl = SparseSdpaKernel::Create(context, node, composite_params, node_index,
                                    threadpool);
  } else if (std::strcmp(composite_params->name, kIndexerUpdateName) == 0) {
    impl = IndexerUpdateKernel::Create(context, node, composite_params,
                                       node_index, threadpool);
  }
  if (impl == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<QsaDelegateKernel>(
      new QsaDelegateKernel(std::move(impl)));
}

TfLiteStatus QsaDelegateKernel::Prepare(TfLiteContext* context) {
  return impl_->Prepare(context);
}

TfLiteStatus QsaDelegateKernel::Invoke(TfLiteContext* context) {
  return impl_->Invoke(context);
}

}  // namespace xnnpack
}  // namespace tflite
