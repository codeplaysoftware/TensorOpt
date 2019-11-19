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
#ifndef SRC_COMMON_MODEL_HPP
#define SRC_COMMON_MODEL_HPP

#include <array>
#include <unordered_map>
#include <vector>

#include "common/memory.hpp"
#include "tensoropt/model.hpp"

struct ANeuralNetworksModel {
  struct ConstHostOperand {
    const void* data;  // weak_ptr
    std::size_t length;
  };

  struct ConstDeviceOperand {
    ConstDeviceOperand(const ANeuralNetworksMemory& m, std::size_t o,
                       std::size_t l)
        : memory(m), offset(o), length(l) {}

    ConstDeviceOperand(const ConstDeviceOperand&) = default;
    ConstDeviceOperand(ConstDeviceOperand&&) = default;
    ConstDeviceOperand& operator=(const ConstDeviceOperand&) = default;
    ConstDeviceOperand& operator=(ConstDeviceOperand&&) = default;

    ANeuralNetworksMemory memory;
    std::size_t offset;
    std::size_t length;
  };

  struct Operation {
    ANeuralNetworksOperationType type;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
  };

  using owned_const_host_data = std::vector<uint8_t>;

  std::vector<std::vector<uint32_t>> operands_dimensions;
  std::vector<ANeuralNetworksOperandType> operands;
  std::unordered_map<uint32_t, ConstHostOperand> const_host_operands;
  std::unordered_map<uint32_t, owned_const_host_data> const_host_operands_owned;
  std::unordered_map<uint32_t, ConstDeviceOperand> const_device_operands;
  bool is_supported_ops_filled;
  std::array<bool, ANEURALNETWORKS_OPERATION_COUNT> supported_ops;
  std::vector<Operation> operations;
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;
  bool finished;
};

#endif  // SRC_COMMON_MODEL_HPP
