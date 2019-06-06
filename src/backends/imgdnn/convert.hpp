// Copyright (C) Codeplay Software Limited.
#ifndef SRC_BACKENDS_IMGDNN_CONVERT_HPP
#define SRC_BACKENDS_IMGDNN_CONVERT_HPP

#include "tensoropt/result.hpp"

struct ANeuralNetworksCompilation;

/**
 * Convert the TensorOpt model to an imgdnn network.
 * Fill imgdnn_network_, imgdnn_inputs_, imgdnn_outputs_
 */
ResultCode convertModel(ANeuralNetworksCompilation* compilation);

#endif  // SRC_BACKENDS_IMGDNN_CONVERT_HPP
