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
#ifndef INCLUDE_TENSOROPT_MODEL_HPP
#define INCLUDE_TENSOROPT_MODEL_HPP

#include "tensoropt/device.hpp"
#include "tensoropt/memory.hpp"
#include "tensoropt/operand.hpp"
#include "tensoropt/operation.hpp"
#include "tensoropt/result.hpp"

struct ANeuralNetworksModel;

/**
 * Create a model.
 */
ResultCode ANeuralNetworksModel_create(ANeuralNetworksModel** model);

/**
 * Mark the model as finished to be able to compile it.
 */
ResultCode ANeuralNetworksModel_finish(ANeuralNetworksModel* model);

/**
 * Free a model.
 * A model cannot be free'd while it is used by a compilation.
 */
void ANeuralNetworksModel_free(ANeuralNetworksModel* model);

/***************
 *   Operand   *
 ***************/

/**
 * Add an operand to the model and fill the operand index if not nullptr.
 */
ResultCode ANeuralNetworksModel_addOperand(
    ANeuralNetworksModel* model, const ANeuralNetworksOperandType* type,
    uint32_t* operand_index = nullptr);

/**
 * Return the number of operands added.
 */
uint32_t ANeuralNetworksModel_getOperandCount(
    const ANeuralNetworksModel* model);

/**
 * Get the ANeuralNetworksOperandType at a given index.
 */
ResultCode ANeuralNetworksModel_getOperandType(
    const ANeuralNetworksModel* model, int32_t index,
    ANeuralNetworksOperandType* type);

/**
 * Set constant operand's value.
 * data is copied to the model if the length is smaller or equal to
 * ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES
 */
ResultCode ANeuralNetworksModel_setOperandValue(ANeuralNetworksModel* model,
                                                int32_t index, const void* data,
                                                std::size_t length);

/**
 * Set constant operand's value from device memory.
 * memory object is always copied to the model.
 */
ResultCode ANeuralNetworksModel_setOperandValueFromMemory(
    ANeuralNetworksModel* model, int32_t index,
    const ANeuralNetworksMemory* memory, std::size_t offset,
    std::size_t length);

/*****************
 *   Operation   *
 *****************/

/**
 * Get the supported operations for a model.
 * The set of devices can be empty but the output is not guarranted to be
 * correct in that case.
 * supported_ops must be at least of size
 * ANeuralNetworksOperationType::ANEURAL_NETWORKS_OPERATION_COUNT.
 */
ResultCode ANeuralNetworksModel_getSupportedOperationsForDevices(
    const ANeuralNetworksModel* model,
    const ANeuralNetworksDevice* const* devices, uint32_t num_devices,
    bool* supported_ops);

/**
 * Return true if a model can add the given operation.
 * The set of devices can be empty but the output is not guarranted to be
 * correct in that case.
 * It will always return false if the operation is not supported by the model
 * and devices. However if the operation is supported it could return false
 * depending on the current state of the model.
 */
bool ANeuralNetworksModel_canAddOperation(
    const ANeuralNetworksModel* model,
    const ANeuralNetworksDevice* const* devices, uint32_t num_devices,
    ANeuralNetworksOperationType op);

/**
 * Add an operation.
 * inputs and outputs are arrays of indices representing operands.
 */
ResultCode ANeuralNetworksModel_addOperation(ANeuralNetworksModel* model,
                                             ANeuralNetworksOperationType op,
                                             uint32_t input_count,
                                             const uint32_t* inputs,
                                             uint32_t output_count,
                                             const uint32_t* outputs);

/**
 * Return the number of operations added.
 */
uint32_t ANeuralNetworksModel_getOperationCount(
    const ANeuralNetworksModel* model);

/**
 * Get the ANeuralNetworksOperationType of an operation.
 */
ResultCode ANeuralNetworksModel_getOperationType(
    const ANeuralNetworksModel* model, int32_t index,
    ANeuralNetworksOperationType* op);

/**
 * Get the number of inputs of an operation.
 */
ResultCode ANeuralNetworksModel_getOperationInputCount(
    const ANeuralNetworksModel* model, int32_t index, uint32_t* input_count);

/**
 * Get the input indices of an operation.
 * Memory is owned by the model.
 */
ResultCode ANeuralNetworksModel_getOperationInputs(
    const ANeuralNetworksModel* model, int32_t index, const uint32_t** inputs);

/**
 * Get the number of outputs of an operation.
 */
ResultCode ANeuralNetworksModel_getOperationOutputCount(
    const ANeuralNetworksModel* model, int32_t index, uint32_t* output_count);

/**
 * Get the output indices of an operation.
 * Memory is owned by the model.
 */
ResultCode ANeuralNetworksModel_getOperationOutputs(
    const ANeuralNetworksModel* model, int32_t index, const uint32_t** outputs);

/****************
 *   Identify   *
 ****************/

/**
 * Set which operands are inputs of the model.
 * This will override any previous identified inputs.
 */
ResultCode ANeuralNetworksModel_identifyInputs(ANeuralNetworksModel* model,
                                               uint32_t input_count,
                                               const uint32_t* inputs);

/**
 * Set which operands are outputs of the model.
 * This will override any previous identified outputs.
 */
ResultCode ANeuralNetworksModel_identifyOutputs(ANeuralNetworksModel* model,
                                                uint32_t output_count,
                                                const uint32_t* outputs);

/**
 * Set which operands are inputs or outputs of the model.
 * This will override any previous identified inputs or outputs.
 */
ResultCode ANeuralNetworksModel_identifyInputsAndOutputs(
    ANeuralNetworksModel* model, uint32_t input_count, const uint32_t* inputs,
    uint32_t output_count, const uint32_t* outputs);

/**
 * Return the number of identified inputs.
 */
uint32_t ANeuralNetworksModel_getIdentifiedInputCount(
    const ANeuralNetworksModel* model);

/**
 * Return the identified input indices.
 */
const uint32_t* ANeuralNetworksModel_getIdentifiedInputs(
    const ANeuralNetworksModel* model);

/**
 * Return the number of identified outputs.
 */
uint32_t ANeuralNetworksModel_getIdentifiedOutputCount(
    const ANeuralNetworksModel* model);

/**
 * Return the identified output indices.
 */
const uint32_t* ANeuralNetworksModel_getIdentifiedOutputs(
    const ANeuralNetworksModel* model);

#endif  // INCLUDE_TENSOROPT_MODEL_HPP
