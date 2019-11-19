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
