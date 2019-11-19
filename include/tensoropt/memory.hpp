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
#ifndef INCLUDE_TENSOROPT_MEMORY_HPP
#define INCLUDE_TENSOROPT_MEMORY_HPP

#include <SYCL/sycl.hpp>
#include <cstddef>

#include "tensoropt/result.hpp"

/**
 * Opaque type representing device memory.
 */
struct ANeuralNetworksMemory;

using tensoropt_buffer_t = cl::sycl::buffer<uint8_t, 1>;

/**
 * Create device Memory from a valid file descriptor using mmap.
 * Not supported on Windows, use ANeuralNetworksMemory_createFromHost instead.
 * protect must be a valid prot argument for the mmap function.
 */
ResultCode ANeuralNetworksMemory_createFromFd(std::size_t size, int protect,
                                              int fd, std::size_t offset,
                                              ANeuralNetworksMemory** memory);

/**
 * Create device Memory from a host buffer.
 * The memory is copied synchronously.
 */
ResultCode ANeuralNetworksMemory_createFromHost(const void* data,
                                                std::size_t size,
                                                ANeuralNetworksMemory** memory);

/**
 * Create device Memory from a SYCL buffer.
 */
ResultCode ANeuralNetworksMemory_createFromBuffer(
    const tensoropt_buffer_t& buffer, ANeuralNetworksMemory** memory);

/**
 * Update the underlying buffer object.
 * Use this method to avoid creating and destroying memory objects.
 */
ResultCode ANeuralNetworksMemory_resetBuffer(ANeuralNetworksMemory* memory,
                                             const tensoropt_buffer_t& buffer);

/**
 * Free the memory. Note that the memory on the device may not be free'd while
 * it is used elsewhere.
 */
void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory);

#endif  // INCLUDE_TENSOROPT_MEMORY_HPP
