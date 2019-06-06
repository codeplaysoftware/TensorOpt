// Copyright (C) Codeplay Software Limited.
#ifndef SRC_COMMON_UTILS_HPP
#define SRC_COMMON_UTILS_HPP

#include <cstdint>
#include <sstream>

#include "tensoropt/operand.hpp"

uint32_t getOperandCodeSizeBytes(ANeuralNetworksOperandCode code);

uint32_t getOperandTypeSize(const ANeuralNetworksOperandType& op);

inline uint32_t getOperandTypeSizeBytes(const ANeuralNetworksOperandType& op) {
  return getOperandTypeSize(op) * getOperandCodeSizeBytes(op.type);
}

template <class Index>
inline Index roundRatioUp(Index x, Index y) {
  return (x + y - 1) / y;
}

template <class T>
std::string arrayToString(const T& data, std::size_t count,
                          std::size_t max_count_print = 10) {
  std::stringstream ss;
  if (count > 0) {
    ss << data[0];
    for (std::size_t i = 1; i < std::min(count, max_count_print); ++i) {
      ss << ", " << data[i];
    }
    if (count > max_count_print) {
      ss << ", ...";
    }
  }
  return ss.str();
}

#endif  // SRC_COMMON_UTILS_HPP
