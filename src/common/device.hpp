// Copyright (C) Codeplay Software Limited.
#ifndef SRC_COMMON_DEVICE_HPP
#define SRC_COMMON_DEVICE_HPP

#include <memory>

#include "tensoropt/device.hpp"

struct ANeuralNetworksDevice {
  cl::sycl::queue* queue;  // weak_ptr
  std::shared_ptr<cl::sycl::queue> owned_queue;
  std::string name;
  std::string version;
  DeviceTypeCode type;
};

#endif  // SRC_COMMON_DEVICE_HPP
