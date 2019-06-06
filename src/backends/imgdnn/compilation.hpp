// Copyright (C) Codeplay Software Limited.
#ifndef SRC_BACKENDS_IMGDNN_COMPILATION_HPP
#define SRC_BACKENDS_IMGDNN_COMPILATION_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/imgdnn/backend.hpp"
#include "common/model.hpp"
#include "tensoropt/compilation.hpp"

struct ANeuralNetworksCompilation {
  const ANeuralNetworksModel* model;    // weak_ptr
  const ANeuralNetworksDevice* device;  // weak_ptr
  std::shared_ptr<ANeuralNetworksDevice> owned_device;
  std::string token_path;
  std::vector<char> cached_file;  // data has to be mutable for IMGDNN API
  bool finished;
  bool serialized;

  using owned_const_host_operands =
      std::unordered_map<uint32_t, ANeuralNetworksModel::owned_const_host_data>;

  // const_copied_to_host_operands is the model's const_device_operand map where
  // each operand was copied to the host. This will be filled by convertModel.
  // The map has to stay alive as long as the ANeuralNetworksCompilation object
  // if the user compiles the same model multiple times.
  owned_const_host_operands const_copied_to_host_operands;

  // IMGDNN specifics
  imgdnn_device imgdnn_device_;
  imgdnn_context imgdnn_context_;
  imgdnn_network imgdnn_network_;
  std::vector<imgdnn_tensor> imgdnn_inputs_;
  std::vector<imgdnn_tensor> imgdnn_outputs_;
  imgdnn_network_object_flags imgdnn_flags_;
  std::string imgdnn_options_;
  imgdnn_network_binary imgdnn_binary_;
  imgdnn_network_object imgdnn_network_object_;
};

#endif  // SRC_BACKENDS_IMGDNN_COMPILATION_HPP
