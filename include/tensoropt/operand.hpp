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
