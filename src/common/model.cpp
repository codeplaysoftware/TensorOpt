/**
 * Copyright (C) Codeplay Software Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "common/model.hpp"
#include "common/macro.hpp"

#include <utility>

ResultCode ANeuralNetworksModel_create(ANeuralNetworksModel** model) {
  TENSOROPT_RETURN_IF_NULL(model);
  *model = new ANeuralNetworksModel();
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_finish(ANeuralNetworksModel* model) {
  model->finished = true;
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksModel_free(ANeuralNetworksModel* model) {
  if (model) {
    delete model;
  }
}

/***************
 *   Operand   *
 ***************/

ResultCode ANeuralNetworksModel_addOperand(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type,
    uint32_t* operand_index) {
  TENSOROPT_RETURN_IF_FINISHED(model);
  if (operand_index) {
    *operand_index = static_cast<uint32_t>(model->operands.size());
  }
  // Keep the dimensions alive internally
  model->operands_dimensions.emplace_back();
  auto& internal_dims = model->operands_dimensions.back();
  internal_dims.assign(type->dimensions,
                       type->dimensions + type->dimensionCount);
  ANeuralNetworksOperandType internal_type = *type;
  internal_type.dimensions = internal_dims.data();
  model->operands.push_back(internal_type);
  return ANEURALNETWORKS_NO_ERROR;
}

uint32_t ANeuralNetworksModel_getOperandCount(
    const ANeuralNetworksModel* model) {
  return static_cast<uint32_t>(model->operands.size());
}

ResultCode ANeuralNetworksModel_getOperandType(
    const ANeuralNetworksModel* model, int32_t index,
    ANeuralNetworksOperandType* type) {
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  *type = model->operands[uindex];
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model,
                                                int32_t index, const void* data,
                                                std::size_t length) {
  TENSOROPT_RETURN_IF_FINISHED(model);
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  if (length <= ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES) {
    auto typed_data = static_cast<
        const ANeuralNetworksModel::owned_const_host_data::value_type*>(data);
    model->const_host_operands_owned[uindex].assign(typed_data,
                                                    typed_data + length);
    // Replace previous Operand value from host if it was set
    model->const_host_operands.erase(uindex);
  } else {
    auto& const_host_operand = model->const_host_operands[uindex];
    const_host_operand.data = data;
    const_host_operand.length = length;
    // Replace previous Operand value from host if it was set
    model->const_host_operands_owned.erase(uindex);
  }

  // Replace previous Operand value from Memory if it was set
  model->const_device_operands.erase(uindex);
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_setOperandValueFromMemory(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, std::size_t offset,
    std::size_t length) {
  TENSOROPT_RETURN_IF_FINISHED(model);
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  ANeuralNetworksModel::ConstDeviceOperand op(*memory, offset, length);
  // Insert op without calling the deleted default constructor
  auto p = model->const_device_operands.emplace(std::make_pair(uindex, op));
  if (!p.second) {
    p.first->second = op;
  }

  // Replace previous Operand value from host if it was set
  model->const_host_operands.erase(uindex);
  model->const_host_operands_owned.erase(uindex);
  return ANEURALNETWORKS_NO_ERROR;
}

/*****************
 *   Operation   *
 *****************/

/*
 * ANeuralNetworksModel_getSupportedOperationsForDevices and
 * ANeuralNetworksModel_canAddOperation have to de implemented in a
 * backend-specific file.
 */

ResultCode ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                             ANeuralNetworksOperationType op,
                                             uint32_t input_count,
                                             const uint32_t* inputs,
                                             uint32_t output_count,
                                             const uint32_t* outputs) {
  TENSOROPT_RETURN_IF_FINISHED(model);
  model->operations.emplace_back();
  auto& operation = model->operations.back();
  operation.type = op;
  operation.inputs.assign(inputs, inputs + input_count);
  operation.outputs.assign(outputs, outputs + output_count);
  return ANEURALNETWORKS_NO_ERROR;
}

uint32_t ANeuralNetworksModel_getOperationCount(
    const ANeuralNetworksModel* model) {
  return static_cast<uint32_t>(model->operations.size());
}

ResultCode ANeuralNetworksModel_getOperationType(
    const ANeuralNetworksModel* model, int32_t index,
    ANeuralNetworksOperationType* op) {
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  *op = model->operations[uindex].type;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_getOperationInputCount(
    const ANeuralNetworksModel* model, int32_t index, uint32_t* input_count) {
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  *input_count = static_cast<uint32_t>(model->operations[uindex].inputs.size());
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_getOperationInputs(
    const ANeuralNetworksModel* model, int32_t index, const uint32_t** inputs) {
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  *inputs = model->operations[uindex].inputs.data();
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_getOperationOutputCount(
    const ANeuralNetworksModel* model, int32_t index, uint32_t* output_count) {
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  *output_count =
      static_cast<uint32_t>(model->operations[uindex].outputs.size());
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_getOperationOutputs(
    const ANeuralNetworksModel* model, int32_t index,
    const uint32_t** outputs) {
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  *outputs = model->operations[uindex].outputs.data();
  return ANEURALNETWORKS_NO_ERROR;
}

/****************
 *   Identify   *
 ****************/

ResultCode ANeuralNetworksModel_identifyInputs(ANeuralNetworksModel* model,
                                               uint32_t input_count,
                                               const uint32_t* inputs) {
  TENSOROPT_RETURN_IF_FINISHED(model);
  model->inputs.assign(inputs, inputs + input_count);
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_identifyOutputs(ANeuralNetworksModel* model,
                                                uint32_t output_count,
                                                const uint32_t* outputs) {
  TENSOROPT_RETURN_IF_FINISHED(model);
  model->outputs.assign(outputs, outputs + output_count);
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t input_count, const uint32_t* inputs,
    uint32_t output_count, const uint32_t* outputs) {
  TENSOROPT_RETURN_IF_FINISHED(model);
  TENSOROPT_RETURN_IF_ERROR(
      ANeuralNetworksModel_identifyInputs(model, input_count, inputs));
  TENSOROPT_RETURN_IF_ERROR(
      ANeuralNetworksModel_identifyOutputs(model, output_count, outputs));
  return ANEURALNETWORKS_NO_ERROR;
}

uint32_t ANeuralNetworksModel_getIdentifiedInputCount(
    const ANeuralNetworksModel* model) {
  return static_cast<uint32_t>(model->inputs.size());
}

const uint32_t* ANeuralNetworksModel_getIdentifiedInputs(
    const ANeuralNetworksModel* model) {
  return model->inputs.data();
}

uint32_t ANeuralNetworksModel_getIdentifiedOutputCount(
    const ANeuralNetworksModel* model) {
  return static_cast<uint32_t>(model->outputs.size());
}

const uint32_t* ANeuralNetworksModel_getIdentifiedOutputs(
    const ANeuralNetworksModel* model) {
  return model->outputs.data();
}
