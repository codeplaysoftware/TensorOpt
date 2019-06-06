// Copyright (C) Codeplay Software Limited.
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
