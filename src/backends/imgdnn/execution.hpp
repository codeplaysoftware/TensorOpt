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
#ifndef SRC_BACKENDS_IMGDNN_EXECUTION_HPP
#define SRC_BACKENDS_IMGDNN_EXECUTION_HPP

#include <map>
#include <mutex>
#include <vector>

#include "backends/imgdnn/backend.hpp"
#include "tensoropt/execution.hpp"

struct ANeuralNetworksExecution {
  struct IdentifiedMemory {
    IdentifiedMemory() = default;
    IdentifiedMemory(ANeuralNetworksMemory* m, std::size_t o, std::size_t l)
        : memory(m), offset(o), length(l) {}
    IdentifiedMemory(const IdentifiedMemory&) = default;
    IdentifiedMemory(IdentifiedMemory&&) = default;
    IdentifiedMemory& operator=(const IdentifiedMemory&) = default;
    IdentifiedMemory& operator=(IdentifiedMemory&&) = default;

    ANeuralNetworksMemory* memory;  // weak_ptr
    std::size_t offset;
    std::size_t length;
  };

  struct HostOutputMemory {
    HostOutputMemory() = default;
    HostOutputMemory(void* d, imgdnn_memory m) : data(d), img_mem(m) {}
    HostOutputMemory(const HostOutputMemory&) = default;
    HostOutputMemory(HostOutputMemory&&) = default;
    HostOutputMemory& operator=(const HostOutputMemory&) = default;
    HostOutputMemory& operator=(HostOutputMemory&&) = default;

    void* data;  // For debug purposes
    imgdnn_memory img_mem;
  };

  bool created_from_compilation;
  const ANeuralNetworksDevice* device;  // weak_ptr

  std::map<uint32_t, IdentifiedMemory> identified_memory_inputs;
  std::map<uint32_t, IdentifiedMemory> identified_memory_outputs;
  std::mutex identified_memory_mutex;
  std::unique_lock<std::mutex> identified_memory_lock;

  // Set used to keep alive dimensions of ANeuralNetworksOperandType
  std::vector<std::vector<uint32_t>> dimensions;

  // Keep the host output imgdnn_memory objects to perform the actual copy to
  // the host once computed
  std::vector<HostOutputMemory> host_output_memories;

  // Keep alive accessors during the interop_task
  using InputAccT = decltype(std::declval<tensoropt_buffer_t>()
                                 .get_access<cl::sycl::access::mode::read>(
                                     std::declval<cl::sycl::handler&>()));
  using OutputAccT = decltype(std::declval<tensoropt_buffer_t>()
                                  .get_access<cl::sycl::access::mode::write>(
                                      std::declval<cl::sycl::handler&>()));
  std::vector<std::pair<uint32_t, InputAccT>> input_indexed_accessors;
  std::vector<std::pair<uint32_t, OutputAccT>> output_indexed_accessors;

  // IMGDNN specifics
  imgdnn_network_object imgdnn_network_object_;
  imgdnn_device imgdnn_device_;
  imgdnn_context imgdnn_context_;
  imgdnn_binding imgdnn_binding_;
  std::vector<imgdnn_input> imgdnn_inputs_;
  std::vector<imgdnn_output> imgdnn_outputs_;
};

#endif  // SRC_BACKENDS_IMGDNN_EXECUTION_HPP
