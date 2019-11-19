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
#include "common/device.hpp"
#include "common/macro.hpp"

ResultCode ANeuralNetworks_getDevice(uint32_t, ANeuralNetworksDevice** device) {
  auto owned_queue = std::make_shared<cl::sycl::queue>();
  TENSOROPT_RETURN_IF_ERROR(
      ANeuralNetworksDevice_create(owned_queue.get(), true, device));
  (*device)->owned_queue = owned_queue;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksDevice_create(cl::sycl::queue* queue, bool get_info,
                                        ANeuralNetworksDevice** device) {
  TENSOROPT_RETURN_IF_NULL(queue);
  TENSOROPT_RETURN_IF_NULL(device);
  auto& deref_device = *device;
  deref_device = new ANeuralNetworksDevice();
  deref_device->queue = queue;
  if (get_info) {
    namespace info = cl::sycl::info;
    deref_device->name =
        deref_device->queue->get_device().get_info<info::device::name>();
    deref_device->version =
        deref_device->queue->get_device().get_info<info::device::version>();
    auto sycl_type =
        deref_device->queue->get_device().get_info<info::device::device_type>();
    switch (sycl_type) {
      case info::device_type::cpu:
        deref_device->type = ANEURALNETWORKS_DEVICE_CPU;
        break;
      case info::device_type::gpu:
        deref_device->type = ANEURALNETWORKS_DEVICE_GPU;
        break;
      case info::device_type::accelerator:
        deref_device->type = ANEURALNETWORKS_DEVICE_ACCELERATOR;
        break;
      default:
        deref_device->type = ANEURALNETWORKS_DEVICE_OTHER;
        break;
    }
  }
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworks_getDeviceCount(uint32_t* num_devices) {
  *num_devices = 1;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksDevice_getFeatureLevel(
    const ANeuralNetworksDevice* device, int64_t* feature_level) {
  TENSOROPT_UNUSED_VARIABLE(device);
  TENSOROPT_RETURN_IF_NULL(feature_level);
  *feature_level = 29;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksDevice_getName(const ANeuralNetworksDevice* device,
                                         const char** name) {
  TENSOROPT_RETURN_IF_NULL(name);
  *name = device->name.c_str();
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksDevice_getType(const ANeuralNetworksDevice* device,
                                         DeviceTypeCode* type) {
  TENSOROPT_RETURN_IF_NULL(type);
  *type = device->type;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksDevice_getVersion(const ANeuralNetworksDevice* device,
                                            const char** version) {
  TENSOROPT_RETURN_IF_NULL(version);
  *version = device->version.c_str();
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksDevice_free(ANeuralNetworksDevice* device) {
  if (device) {
    delete device;
  }
}
