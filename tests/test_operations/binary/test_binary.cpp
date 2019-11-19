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

#include <array>

class BinaryFixture : public CommonFixture {
 protected:
  template <std::size_t N = 2>
  void setBinaryInputsAndOutputs(ANeuralNetworksOperationType op_type,
                                 const std::array<uint32_t, N>& op_inputs_idx,
                                 uint32_t op_output_idx) {
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_addOperation(
        model, op_type, N, op_inputs_idx.data(), 1, &op_output_idx));

    // Model's inputs are the first 2 operation's inputs
    // Model's output is the operation's output
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_identifyInputsAndOutputs(
        model, 2, op_inputs_idx.data(), 1, &op_output_idx));
  }

  template <std::size_t N = 2>
  inline void runBinaryOperation(ANeuralNetworksOperationType op_type,
                                 const std::array<uint32_t, N>& op_inputs_idx,
                                 uint32_t op_output_idx) {
    setBinaryInputsAndOutputs(op_type, op_inputs_idx, op_output_idx);
    compileAndExecute();
  }

  void testHighAndHighRank(ANeuralNetworksOperationType op_type) {
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {1, 1, 2, 3, 4, 5});  // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {1, 5, 1, 1, 1, 1});  // 1
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {1, 5, 2, 3, 4, 5});  // 2

    runBinaryOperation(op_type, {0, 1}, 2);
  }

  void testLowAndHighRank(ANeuralNetworksOperationType op_type) {
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {5, 7});           // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {2, 3, 1, 5, 1});  // 1
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {2, 3, 1, 5, 7});  // 2

    runBinaryOperation(op_type, {0, 1}, 2);
  }

  void testScalarAndHighRank(ANeuralNetworksOperationType op_type) {
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);                // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {2, 3, 5, 1});  // 1
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {2, 3, 5, 1});  // 2

    runBinaryOperation(op_type, {0, 1}, 2);
  }

  void testScalarAndVecOne(ANeuralNetworksOperationType op_type) {
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);       // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {1});  // 1
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);       // 2

    runBinaryOperation(op_type, {0, 1}, 2);
  }

  void testScalarAndScalar(ANeuralNetworksOperationType op_type) {
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);  // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);  // 1
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);  // 2

    runBinaryOperation(op_type, {0, 1}, 2);
  }

  void testScalarAndScalarWithRelu(ANeuralNetworksOperationType op_type) {
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);  // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);  // 1
    addConstScalarOperand(ANEURALNETWORKS_TENSOR_INT32,
                          ANEURALNETWORKS_FUSED_RELU);  // 2
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);         // 3

    runBinaryOperation<3>(op_type, {0, 1, 2}, 3);
  }

  void testCheckValidOutput(ANeuralNetworksOperationType op_type) {
    // Keep the test that check for valid output very simple.
    // The first input will always be scalar to avoid any complex reshape.
    float host_input0{1.f};
    std::vector<float> host_input1{-1.f, 2.f, 5.f};
    uint32_t input1_size = static_cast<uint32_t>(host_input1.size());
    std::vector<float> host_output(input1_size);

    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32);                 // 0
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {input1_size});  // 1
    addOperand(ANEURALNETWORKS_TENSOR_FLOAT32, {input1_size});  // 2

    setBinaryInputsAndOutputs(op_type, {0, 1}, 2);
    compileModel();

    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_setInput(
        execution, 0, nullptr, &host_input0, sizeof(float)));
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_setInput(
        execution, 1, nullptr, host_input1.data(),
        input1_size * sizeof(float)));
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_setOutput(
        execution, 0, nullptr, host_output.data(),
        input1_size * sizeof(float)));
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_compute(execution));

    std::function<float(float, float)> host_functor;
    switch (op_type) {
      case ANEURALNETWORKS_ADD:
        host_functor = std::plus<float>();
        break;

      case ANEURALNETWORKS_MUL:
        host_functor = std::multiplies<float>();
        break;

      case ANEURALNETWORKS_SUB:
        host_functor = std::minus<float>();
        break;

      case ANEURALNETWORKS_DIV:
        host_functor = std::divides<float>();
        break;

      case ANEURALNETWORKS_MAX:
        host_functor = [](float lhs, float rhs) { return std::max(lhs, rhs); };
        break;

      case ANEURALNETWORKS_MIN:
        host_functor = [](float lhs, float rhs) { return std::min(lhs, rhs); };
        break;

      default:
        FAIL() << "Test does not support operation " << op_type;
        break;
    }

    for (uint32_t i = 0; i < input1_size; ++i) {
      ASSERT_FLOAT_EQ(host_output[i],
                      host_functor(host_input0, host_input1[i]));
    }
  }
};

#define ADD_BINARY_TEST_HELPER(OP, NAME, NN_OP) \
  ADD_TEST_HELPER_ARGS(BinaryFixture, OP##NAME, test##NAME, NN_OP)

#define ADD_BINARY_TEST(OP, NN_OP)                           \
  ADD_BINARY_TEST_HELPER(OP, HighAndHighRank, NN_OP)         \
  ADD_BINARY_TEST_HELPER(OP, LowAndHighRank, NN_OP)          \
  ADD_BINARY_TEST_HELPER(OP, ScalarAndHighRank, NN_OP)       \
  ADD_BINARY_TEST_HELPER(OP, ScalarAndVecOne, NN_OP)         \
  ADD_BINARY_TEST_HELPER(OP, ScalarAndScalar, NN_OP)         \
  ADD_BINARY_TEST_HELPER(OP, ScalarAndScalarWithRelu, NN_OP) \
  ADD_BINARY_TEST_HELPER(OP, CheckValidOutput, NN_OP)

ADD_BINARY_TEST(add, ANEURALNETWORKS_ADD)
ADD_BINARY_TEST(mul, ANEURALNETWORKS_MUL)
ADD_BINARY_TEST(sub, ANEURALNETWORKS_SUB)
ADD_BINARY_TEST(div, ANEURALNETWORKS_DIV)
ADD_BINARY_TEST(max, ANEURALNETWORKS_MAX)
ADD_BINARY_TEST(min, ANEURALNETWORKS_MIN)
