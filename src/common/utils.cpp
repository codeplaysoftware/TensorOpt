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
#include "common/utils.hpp"

uint32_t getOperandCodeSizeBytes(ANeuralNetworksOperandCode code) {
  switch (code) {
    case ANEURALNETWORKS_BOOL:
    case ANEURALNETWORKS_TENSOR_BOOL8:
      return 1;

    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_UINT32:
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_INT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      return 4;

    default:
      return 0;
  }
}

uint32_t getOperandTypeSize(const ANeuralNetworksOperandType& op) {
  uint32_t size = 1;
  for (unsigned i = 0; i < op.dimensionCount; ++i) {
    size *= op.dimensions[i];
  }
  return size;
}
