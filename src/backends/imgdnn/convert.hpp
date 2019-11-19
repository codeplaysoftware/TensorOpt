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
