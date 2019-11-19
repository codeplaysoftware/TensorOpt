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
#include "common/model.hpp"
#include "common/macro.hpp"

ResultCode ANeuralNetworksModel_getSupportedOperationsForDevices(
    const ANeuralNetworksModel* model,
    const ANeuralNetworksDevice* const* devices, uint32_t num_devices,
    bool* supported_ops) {
  TENSOROPT_UNUSED_VARIABLE(model);
  TENSOROPT_UNUSED_VARIABLE(devices);
  TENSOROPT_UNUSED_VARIABLE(num_devices);
  TENSOROPT_RETURN_IF_NULL(supported_ops);
  for (unsigned i = 0; i < ANEURALNETWORKS_OPERATION_COUNT; ++i) {
    supported_ops[i] = true;
  }
  return ANEURALNETWORKS_NO_ERROR;
}

bool ANeuralNetworksModel_canAddOperation(
    const ANeuralNetworksModel* model,
    const ANeuralNetworksDevice* const* devices, uint32_t num_devices,
    ANeuralNetworksOperationType op) {
  static bool supported_ops[ANEURALNETWORKS_OPERATION_COUNT];
  static bool initialized = false;
  if (!initialized) {
    // model, devices and num_devices does not affect the result for this
    // backend.
    TENSOROPT_RETURN_IF_ERROR(
        ANeuralNetworksModel_getSupportedOperationsForDevices(
            model, devices, num_devices, supported_ops));
    initialized = true;
  }
  return supported_ops[op];
}
