// Copyright (C) Codeplay Software Limited.
#ifndef INCLUDE_TENSOROPT_OPERAND_HPP
#define INCLUDE_TENSOROPT_OPERAND_HPP

#include <cstddef>
#include <cstdint>

/**
 * Host operands smaller or equal to this size will be immediately copied.
 * Otherwise it is up to the user to make sure the memory is still
 * available until the compilation for constant host operands or the execution
 * for identified host inputs.
 */
enum { ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES = 128 };

/**
 * An operand represents a type of memory that can be added to a model
 * and attached to an operation as an input or output.
 */
enum ANeuralNetworksOperandCode : int {
  // Scalar types (host memory)
  ANEURALNETWORKS_BOOL,
  ANEURALNETWORKS_INT32,
  ANEURALNETWORKS_UINT32,
  ANEURALNETWORKS_FLOAT32,

  // Tensor types (device memory)
  ANEURALNETWORKS_TENSOR_BOOL8,
  ANEURALNETWORKS_TENSOR_INT32,
  ANEURALNETWORKS_TENSOR_FLOAT32,

  ANEURALNETWORKS_INVALID,
};

/**
 * Describes an operand.
 */
struct ANeuralNetworksOperandType {
  /** Data type of the operand */
  ANeuralNetworksOperandCode type;
  /** Number of dimensions (rank), should be 0 for scalars */
  uint32_t dimensionCount;
  /** The dimensions of the tensor, should be nullptr for scalars */
  const uint32_t* dimensions;
  /** Used for quantized type */
  float scale;
  /** Used for quantized type */
  int32_t zeroPoint;
};

#endif  // INCLUDE_TENSOROPT_OPERAND_HPP
