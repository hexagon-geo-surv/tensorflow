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

#include "tensorflow/lite/experimental/lrt/core/model.h"

#include <cstddef>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_options.h"

//
// Model
//

LrtStatus GetModelNumSubgraphs(LrtModel model,
                               lrt_param_index_t* num_subgraphs) {
  *num_subgraphs = model->subgraphs.size();
  return kLrtStatusOk;
}

LrtStatus GetModelSubgraph(LrtModel model, lrt_param_index_t subgraph_index,
                           LrtSubgraph* subgraph) {
  if (subgraph_index >= model->subgraphs.size()) {
    return kLrtStatusParamIndexOOB;
  }
  *subgraph = model->subgraphs.data() + subgraph_index;
  return kLrtStatusOk;
}

LrtStatus GetModelMainSubgraph(LrtModel model,
                               lrt_param_index_t* main_subgraph_index) {
  // TODO replace this with signature.
  *main_subgraph_index = 0;
  return kLrtStatusOk;
}

void ModelDestroy(LrtModel model) { delete model; }

LrtStatus PushOp(LrtOpList op_list, LrtOp op) {
  op_list->ops.push_back(op);
  return kLrtStatusOk;
}

//
// Subgraph
//

LrtStatus GetSubgraphInputs(LrtSubgraph subgraph, lrt_param_index_t* num_inputs,
                            LrtTensorArray* inputs) {
  *num_inputs = subgraph->inputs.size();
  *inputs = subgraph->inputs.data();
  return kLrtStatusOk;
}

LrtStatus GetSubgraphOutputs(LrtSubgraph subgraph,
                             lrt_param_index_t* num_outputs,
                             LrtTensorArray* outputs) {
  *num_outputs = subgraph->outputs.size();
  *outputs = subgraph->outputs.data();
  return kLrtStatusOk;
}

LrtStatus GetSubgraphOps(LrtSubgraph subgraph, lrt_param_index_t* num_ops,
                         LrtOpArray* ops) {
  *num_ops = subgraph->ops.size();
  *ops = subgraph->ops.data();
  return kLrtStatusOk;
}

//
// Op
//

LrtStatus GetOpOutputs(LrtOp op, lrt_param_index_t* num_outputs,
                       LrtTensorArray* outputs) {
  *num_outputs = op->outputs.size();
  *outputs = op->outputs.data();
  return kLrtStatusOk;
}

LrtStatus GetOpInputs(LrtOp op, lrt_param_index_t* num_inputs,
                      LrtTensorArray* inputs) {
  *num_inputs = op->inputs.size();
  *inputs = op->inputs.data();
  return kLrtStatusOk;
}

LrtStatus GetOpCode(LrtOp op, LrtOpCode* code) {
  *code = op->op_code;
  return kLrtStatusOk;
}

LrtStatus LrtOpGetFusedActivationOption(
    LrtOp op, LrtFusedActivationOption* fused_activation) {
  switch (op->op_code) {
    case kLrtOpCodeTflAdd:
      *fused_activation = op->option.AsAddOptions()->fused_activation_function;
      break;
    case kLrtOpCodeTflConcatenation:
      *fused_activation =
          op->option.AsConcatenationOptions()->fused_activation_function;
      break;
    case kLrtOpCodeTflDiv:
      *fused_activation = op->option.AsDivOptions()->fused_activation_function;
      break;
    case kLrtOpCodeTflFullyConnected:
      *fused_activation =
          op->option.AsFullyConnectedOptions()->fused_activation_function;
      break;
    case kLrtOpCodeTflMul:
      *fused_activation = op->option.AsMulOptions()->fused_activation_function;
      break;
    case kLrtOpCodeTflSub:
      *fused_activation = op->option.AsSubOptions()->fused_activation_function;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetAxisOption(LrtOp op, LrtAxisOption* axis) {
  switch (op->op_code) {
    case kLrtOpCodeTflConcatenation:
      *axis = op->option.AsConcatenationOptions()->axis;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetAdjXOption(LrtOp op, LrtAdjXOption* adj_x) {
  switch (op->op_code) {
    case kLrtOpCodeTflBatchMatmul:
      *adj_x = op->option.AsBatchMatMulOptions()->adj_x;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetAdjYOption(LrtOp op, LrtAdjYOption* adj_y) {
  switch (op->op_code) {
    case kLrtOpCodeTflBatchMatmul:
      *adj_y = op->option.AsBatchMatMulOptions()->adj_y;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetAsymmetricQuantizeInputOption(
    LrtOp op, LrtAsymmetricQuantizeInputOption* asymmetric_quantize_input) {
  switch (op->op_code) {
    case kLrtOpCodeTflBatchMatmul:
      *asymmetric_quantize_input =
          op->option.AsBatchMatMulOptions()->asymmetric_quantize_inputs;
      break;
    case kLrtOpCodeTflFullyConnected:
      *asymmetric_quantize_input =
          op->option.AsFullyConnectedOptions()->asymmetric_quantize_inputs;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetWeightsFormatOption(LrtOp op,
                                      LrtWeightsFormatOption* weights_format) {
  switch (op->op_code) {
    case kLrtOpCodeTflFullyConnected:
      *weights_format = op->option.AsFullyConnectedOptions()->weights_format;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetKeepNumDimsOption(LrtOp op,
                                    LrtKeepNumDimsOption* keep_num_dims) {
  switch (op->op_code) {
    case kLrtOpCodeTflFullyConnected:
      *keep_num_dims = op->option.AsFullyConnectedOptions()->keep_num_dims;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetQuantizedBiasTypeOption(
    LrtOp op, LrtQuantizedBiasTypeOption* quantized_bias_type) {
  switch (op->op_code) {
    case kLrtOpCodeTflFullyConnected:
      *quantized_bias_type =
          op->option.AsFullyConnectedOptions()->quantized_bias_type;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetBetaOption(LrtOp op, LrtBetaOption* beta) {
  switch (op->op_code) {
    case kLrtOpCodeTflSoftmax:
      *beta = op->option.AsSoftmaxOptions()->beta;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetBeginMaskOption(LrtOp op, LrtBeginMaskOption* begin_mask) {
  switch (op->op_code) {
    case kLrtOpCodeTflStridedSlice:
      *begin_mask = op->option.AsStridedSliceOptions()->begin_mask;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetEndMaskOption(LrtOp op, LrtEndMaskOption* end_mask) {
  switch (op->op_code) {
    case kLrtOpCodeTflStridedSlice:
      *end_mask = op->option.AsStridedSliceOptions()->end_mask;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetEllipsisMaskOption(LrtOp op,
                                     LrtEllipsisMaskOption* ellipsis_mask) {
  switch (op->op_code) {
    case kLrtOpCodeTflStridedSlice:
      *ellipsis_mask = op->option.AsStridedSliceOptions()->ellipsis_mask;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetNewAxisMaskOption(LrtOp op,
                                    LrtNewAxisMaskOption* new_axis_mask) {
  switch (op->op_code) {
    case kLrtOpCodeTflStridedSlice:
      *new_axis_mask = op->option.AsStridedSliceOptions()->new_axis_mask;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetShrinkAxisMaskOption(
    LrtOp op, LrtShrinkAxisMaskOption* shrink_axis_mask) {
  switch (op->op_code) {
    case kLrtOpCodeTflStridedSlice:
      *shrink_axis_mask = op->option.AsStridedSliceOptions()->shrink_axis_mask;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetOffsetOption(LrtOp op, LrtOffsetOption* offset) {
  switch (op->op_code) {
    case kLrtOpCodeTflStridedSlice:
      *offset = op->option.AsStridedSliceOptions()->offset;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

LrtStatus LrtOpGetPotScaleInt16Option(LrtOp op,
                                      LrtPotScaleInt16Option* pot_scale_int16) {
  switch (op->op_code) {
    case kLrtOpCodeTflSub:
      *pot_scale_int16 = op->option.AsSubOptions()->pot_scale_int16;
      break;
    default:
      return kLrtStatusErrorNotFound;
  }
  return kLrtStatusOk;
}

//
// Tensor
//

LrtStatus GetWeightsInfo(LrtWeights weights, size_t* size, const void** addr) {
  if (weights->fb_buffer == nullptr) {
    *size = 0;
    *addr = nullptr;
  } else {
    *size = weights->fb_buffer->data.size();
    *addr = weights->fb_buffer->data.data();
  }
  return kLrtStatusOk;
}

LrtStatus GetTensorWeights(LrtTensor tensor, LrtWeights* weights) {
  *weights = &tensor->weights;
  return kLrtStatusOk;
}

LrtStatus GetTensorUses(LrtTensor tensor, lrt_param_index_t* num_uses,
                        LrtOpArray* use_users,
                        lrt_param_index_t** use_user_arg_inds) {
  *num_uses = tensor->users.size();
  *use_users = tensor->users.data();
  *use_user_arg_inds = tensor->user_arg_inds.data();
  return kLrtStatusOk;
}

// Null if subgraph input or constant.
LrtStatus GetTensorDefiningOp(LrtTensor tensor, LrtOp* maybe_defining_op,
                              lrt_param_index_t* maybe_defining_op_output_ind) {
  if (tensor->defining_op != nullptr) {
    *maybe_defining_op = tensor->defining_op;
    *maybe_defining_op_output_ind = tensor->defining_op_out_ind;
  }
  return kLrtStatusOk;
}

LrtStatus GetTensorTypeId(LrtTensor tensor, LrtTensorTypeId* type_id) {
  *type_id = tensor->type_id;
  return kLrtStatusOk;
}

LrtStatus GetUrankedTensorType(LrtTensor tensor,
                               LrtUnrankedTensorType* unranked_tensor_type) {
  if (tensor->type_id != kLrtUnrankedTensorType) {
    return kLrtStatusBadTensorType;
  }
  *unranked_tensor_type = tensor->type_detail.unranked_tensor_type;
  return kLrtStatusOk;
}

LrtStatus GetRankedTensorType(LrtTensor tensor,
                              LrtRankedTensorType* ranked_tensor_type) {
  if (tensor->type_id != kLrtRankedTensorType) {
    return kLrtStatusBadTensorType;
  }
  *ranked_tensor_type = tensor->type_detail.ranked_tensor_type;
  return kLrtStatusOk;
}
