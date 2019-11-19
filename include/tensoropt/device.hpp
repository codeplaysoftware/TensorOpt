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
#ifndef INCLUDE_TENSOROPT_DEVICE_HPP
#define INCLUDE_TENSOROPT_DEVICE_HPP

#include <SYCL/sycl.hpp>
#include <cstdint>

#include "tensoropt/result.hpp"

enum DeviceTypeCode : int {
  ANEURALNETWORKS_DEVICE_ACCELERATOR,
  ANEURALNETWORKS_DEVICE_CPU,
  ANEURALNETWORKS_DEVICE_GPU,
  ANEURALNETWORKS_DEVICE_OTHER,
  ANEURALNETWORKS_DEVICE_UNKNOWN,
};

struct ANeuralNetworksDevice;

/**
 * Create a device using a default SYCL queue.
 * device_idx is ignored as it doesn't map to a SYCL device.
 * See ANeuralNetworksDevice_create to create a specific device.
 */
ResultCode ANeuralNetworks_getDevice(uint32_t device_idx,
                                     ANeuralNetworksDevice** device);

/**
 * Create a device from an existing SYCL queue.
 */
ResultCode ANeuralNetworksDevice_create(cl::sycl::queue* queue, bool get_info,
                                        ANeuralNetworksDevice** device);

/**
 * Always set num_devices to 1 as SYCL will always have at least one device
 * available.
 * This function is meant to be used with ANeuralNetworks_getDevice which is
 * not the recommended way of creating a device.
 */
ResultCode ANeuralNetworks_getDeviceCount(uint32_t* num_devices);

/**
 * Get the supported NNAPI version of the specified device.
 * The device must have been created with get_info set to true to get
 * any meaningful information.
 */
ResultCode ANeuralNetworksDevice_getFeatureLevel(
    const ANeuralNetworksDevice* device, int64_t* feature_level);

/**
 * Get the name of the device.
 * The device must have been created with get_info set to true to get
 * any meaningful information.
 */
ResultCode ANeuralNetworksDevice_getName(const ANeuralNetworksDevice* device,
                                         const char** name);

/**
 * Get the type of the device.
 * The device must have been created with get_info set to true to get
 * any meaningful information.
 */
ResultCode ANeuralNetworksDevice_getType(const ANeuralNetworksDevice* device,
                                         DeviceTypeCode* type);

/**
 * Get the version of the device.
 * The device must have been created with get_info set to true to get
 * any meaningful information.
 */
ResultCode ANeuralNetworksDevice_getVersion(const ANeuralNetworksDevice* device,
                                            const char** version);

/**
 * Free a device.
 */
void ANeuralNetworksDevice_free(ANeuralNetworksDevice* device);

#endif  // INCLUDE_TENSOROPT_DEVICE_HPP
