// Copyright (C) Codeplay Software Limited.
#ifndef TENSOROPT_TESTS_COMMON_COMMON_FIXTURE_HPP
#define TENSOROPT_TESTS_COMMON_COMMON_FIXTURE_HPP

#include "common/test_utils.hpp"

class CommonFixture : public ::testing::Test {
 protected:
  CommonFixture() : model(nullptr), compilation(nullptr), execution(nullptr) {
    ANeuralNetworksModel_create(&model);
  }

  CommonFixture(CommonFixture&) = delete;
  CommonFixture(CommonFixture&&) = default;

  CommonFixture& operator=(CommonFixture&) = delete;
  CommonFixture& operator=(CommonFixture&&) = default;

  template <class Container = std::vector<uint32_t>>
  void addOperand(ANeuralNetworksOperandCode op_code,
                  const Container& dimensions = Container(),
                  uint32_t* op_idx = nullptr) {
    ANeuralNetworksOperandType op;
    op.type = op_code;
    op.scale = 0.f;
    op.zeroPoint = 0;
    op.dimensionCount = static_cast<uint32_t>(dimensions.size());
    op.dimensions = dimensions.empty() ? nullptr : dimensions.data();
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_addOperand(model, &op, op_idx));
  }

  template <class T>
  void addConstScalarOperand(ANeuralNetworksOperandCode op_code, T value,
                             uint32_t* op_idx = nullptr) {
    uint32_t local_idx;
    if (!op_idx) {
      op_idx = &local_idx;
    }
    addOperand(op_code, {}, op_idx);
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_setOperandValue(
        model, *op_idx, &value, sizeof(T)));
  }

  void compileModel() {
    TENSOROPT_ASSERT_OK(ANeuralNetworksModel_finish(model));
    TENSOROPT_ASSERT_OK(ANeuralNetworksCompilation_create(model, &compilation));
    TENSOROPT_ASSERT_OK(ANeuralNetworksCompilation_finish(compilation));
    TENSOROPT_ASSERT_OK(
        ANeuralNetworksExecution_create(compilation, &execution));
  }

  // Execute the model with unspecified inputs
  // Use only if the test doesn't check for valid output
  void executeCompilation() {
    auto nb_inputs =
        ANeuralNetworksExecution_getIdentifiedInputCount(execution);
    std::vector<ANeuralNetworksOperandType> op_inputs(nb_inputs);
    std::vector<std::vector<uint8_t>> data_inputs(nb_inputs);
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_getIdentifiedInputs(
        execution, op_inputs.data()));
    for (uint32_t i = 0; i < nb_inputs; ++i) {
      auto& data = data_inputs[i];
      data.resize(getOperandTypeSizeBytes(op_inputs[i]));
      TENSOROPT_ASSERT_OK(
          ANeuralNetworksExecution_setInput(execution, static_cast<int32_t>(i),
                                            nullptr, data.data(), data.size()));
    }

    auto nb_outputs =
        ANeuralNetworksExecution_getIdentifiedOutputCount(execution);
    std::vector<ANeuralNetworksOperandType> op_outputs(nb_inputs);
    std::vector<std::vector<uint8_t>> data_outputs(nb_outputs);
    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_getIdentifiedOutputs(
        execution, op_outputs.data()));
    for (uint32_t i = 0; i < nb_outputs; ++i) {
      auto& data = data_outputs[i];
      data.resize(getOperandTypeSizeBytes(op_outputs[i]));
      TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_setOutput(
          execution, static_cast<int32_t>(i), nullptr, data.data(),
          data.size()));
    }

    TENSOROPT_ASSERT_OK(ANeuralNetworksExecution_compute(execution));
  }

  void compileAndExecute() {
    compileModel();
    executeCompilation();
  }

  virtual ~CommonFixture() {
    if (execution) {
      ANeuralNetworksExecution_free(execution);
    }
    if (compilation) {
      ANeuralNetworksCompilation_free(compilation);
    }
    if (model) {
      ANeuralNetworksModel_free(model);
    }
  }

  ANeuralNetworksModel* model;
  ANeuralNetworksCompilation* compilation;
  ANeuralNetworksExecution* execution;
};

#endif  // TENSOROPT_TESTS_COMMON_COMMON_FIXTURE_HPP
