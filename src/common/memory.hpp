// Copyright (C) Codeplay Software Limited.
#ifndef SRC_COMMON_MEMORY_HPP
#define SRC_COMMON_MEMORY_HPP

#include "tensoropt/memory.hpp"

struct ANeuralNetworksMemory {
  tensoropt_buffer_t buffer;
};

#endif  // SRC_COMMON_MEMORY_HPP
