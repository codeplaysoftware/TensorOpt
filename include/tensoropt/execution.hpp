// Copyright (C) Codeplay Software Limited.
#ifndef INCLUDE_TENSOROPT_EXECUTION_HPP
#define INCLUDE_TENSOROPT_EXECUTION_HPP

#include "tensoropt/compilation.hpp"
#include "tensoropt/event.hpp"

struct ANeuralNetworksExecution;

/**
 * Create an execution from a finished compilation.
 * It will use the same device than for the compilation.
 */
ResultCode ANeuralNetworksExecution_create(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution);

/**
 * Create an execution from serialized data and a device.
 */
ResultCode ANeuralNetworksExecution_createFromBinary(
    const void* data, std::size_t data_size,
    const ANeuralNetworksDevice* device, ANeuralNetworksExecution** execution);

/**
 * Set the value of an identified model input.
 * There must be one call of ANeuralNetworksExecution_setInput or
 * ANeuralNetworksExecution_setInputFromMemory per model input.
 * The index is an identified input index, not an operand index.
 * type can be a nullptr.
 * If the input is optional, value can be nullptr and length can be 0.
 * data is copied to the model if the length is smaller or equal to
 * ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES
 */
ResultCode ANeuralNetworksExecution_setInput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* data,
    std::size_t length);

/**
 * Set the value of an identified model input from device memory.
 * There must be one call of ANeuralNetworksExecution_setInput or
 * ANeuralNetworksExecution_setInputFromMemory per model input.
 * The index is an identified input index, not an operand index.
 * type can be a nullptr.
 * If the input is optional, memory can be nullptr and offset and length can
 * be 0.
 */
ResultCode ANeuralNetworksExecution_setInputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    std::size_t offset, std::size_t length);

/**
 * Set the value of an identified model output.
 * There must be one call of ANeuralNetworksExecution_setOutput or
 * ANeuralNetworksExecution_setOutputFromMemory per model output.
 * The index is an identified output index, not an operand index.
 * type can be a nullptr.
 * If the output is optional, value can be nullptr and length can be 0.
 */
ResultCode ANeuralNetworksExecution_setOutput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* data, std::size_t length);

/**
 * Set the value of an identified model output from device memory.
 * There must be one call of ANeuralNetworksExecution_setOutput or
 * ANeuralNetworksExecution_setOutputFromMemory per model output.
 * The index is an identified output index, not an operand index.
 * type can be a nullptr.
 * If the output is optional, memory can be nullptr and offset and length can
 * be 0.
 */
ResultCode ANeuralNetworksExecution_setOutputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    std::size_t offset, std::size_t length);

/**
 * Return the number of identified inputs.
 * Same as ANeuralNetworksModel_getIdentifiedInputCount.
 */
uint32_t ANeuralNetworksExecution_getIdentifiedInputCount(
    const ANeuralNetworksExecution* execution);

/**
 * Return the identified input operands.
 * Memory representing the dimensions of the inputs is owned by the execution.
 * inputs are invalidated after a call to ANeuralNetworksExecution_compute or
 * ANeuralNetworksExecution_startCompute.
 */
ResultCode ANeuralNetworksExecution_getIdentifiedInputs(
    ANeuralNetworksExecution* execution, ANeuralNetworksOperandType* inputs);

/**
 * Return the number of identified outputs.
 * Same as ANeuralNetworksModel_getIdentifiedOutputCount.
 */
uint32_t ANeuralNetworksExecution_getIdentifiedOutputCount(
    const ANeuralNetworksExecution* execution);

/**
 * Return the identified output operands.
 * Memory representing the dimensions of the outputs is owned by the execution.
 * outputs are invalidated after a call to ANeuralNetworksExecution_compute or
 * ANeuralNetworksExecution_startCompute.
 */
ResultCode ANeuralNetworksExecution_getIdentifiedOutputs(
    ANeuralNetworksExecution* execution, ANeuralNetworksOperandType* outputs);

/**
 * Return the dimensions of the specified output operand.
 * dimensions must be at least as big as the rank of the operand.
 * See also ANeuralNetworksExecution_getIdentifiedOutputs.
 */
ResultCode ANeuralNetworksExecution_getOutputOperandDimensions(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* dimensions);

/**
 * Return the rank of the specified output operand.
 * See also ANeuralNetworksExecution_getIdentifiedOutputs.
 */
ResultCode ANeuralNetworksExecution_getOutputOperandRank(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* rank);

/**
 * Execute the model synchronously.
 * An execution can be executed multiple times.
 */
ResultCode ANeuralNetworksExecution_compute(
    ANeuralNetworksExecution* execution);

/**
 * Execute the model asynchronously and create a corresponding output_event.
 * An execution can be executed multiple times. The output_event can be nullptr
 * as the SYCL runtime will handle the dependencies. The user must wait on
 * output_event before being able to access any host output.
 * The execution must outlive any event that it created.
 */
ResultCode ANeuralNetworksExecution_startCompute(
    ANeuralNetworksExecution* execution, ANeuralNetworksEvent** output_event);

/**
 * Free an execution.
 */
void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution);

#endif  // INCLUDE_TENSOROPT_EXECUTION_HPP
