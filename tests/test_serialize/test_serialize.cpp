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
#include "common/common_fixture.hpp"

class SerializeFixture : public CommonFixture {
 protected:
  inline void addOperation(const std::array<uint32_t, 2>& op_inputs_idx,
                           uint32_t op_output_idx) {
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_addOperation(
        model, ANEURALNETWORKS_ADD, 2, op_inputs_idx.data(), 1,
        &op_output_idx));

    // Model's inputs are the first 2 operation's inputs
    // Model output is the operation's output
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_identifyInputsAndOutputs(
        model, 2, op_inputs_idx.data(), 1, &op_output_idx));
  }

  void testCheckValidOutput() {
    // Keep the test that check for valid output very simple.
    // The first input will always be scalar to avoid any complex reshape.
    ANeuralNetworksDevice* device = nullptr;
    TENSOROPT_ASSERT_OK(ANeuralNetworks_getDevice(0, &device));

    float host_input0{1.f};
    std::vector<float> host_input1{-1.f, 2.f, 5.f};
    uint32_t input1_size = static_cast<uint32_t>(host_input1.size());
    std::vector<float> host_output(input1_size);

    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);                 // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {input1_size});  // 1
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {input1_size});  // 2
    addOperation({0, 1}, 2);

    // Serialize the model
    void* serialize_data;
    std::size_t serialize_size;
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_finish(model));
    TENSOROPT_ASSERT_OK(ANeuralNetworksCompilation_createForDevices(
        model, &device, 1, &compilation));
    TENSOROPT_ASSERT_OK(ANeuralNetworksCompilation_serialize(
        compilation, &serialize_data, &serialize_size));
    ANeuralNetworksCompilation_free(compilation);
    compilation = nullptr;
    ANeuralNetworksModel_free(model);
    model = nullptr;

    // Deserialize the model
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_createFromBinary(
        serialize_data, serialize_size, device, &execution));

    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_setInput(
        execution, 0, nullptr, &host_input0, sizeof(float)));
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_setInput(
        execution, 1, nullptr, host_input1.data(),
        input1_size * sizeof(float)));
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_setOutput(
        execution, 0, nullptr, host_output.data(),
        input1_size * sizeof(float)));
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_compute(execution));
    ANeuralNetworksDevice_free(device);

    auto host_functor = std::plus<float>();
    for (uint32_t i = 0; i < input1_size; ++i) {
      ASSERT_FLOAT_EQ(host_output[i],
                      host_functor(host_input0, host_input1[i]));
    }
  }
};

#define ADD_SERIALIZE_TEST_HELPER(NAME) \
  ADD_TEST_HELPER(SerializeFixture, NAME, test##NAME)

ADD_SERIALIZE_TEST_HELPER(CheckValidOutput)
