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
#include <imgdnn/imgdnn.h>

/*
 * Add symbols missing from libIMGDNN.so in some versions of the DDK.
 */

imgdnn_network_binary __attribute__((weak)) imgdnnCreateNetworkBinary(
    const imgdnn_device, const imgdnn_context, const imgdnn_network,
    unsigned int, const imgdnn_tensor[], unsigned int, const imgdnn_tensor[],
    const imgdnn_network_object_flags, const char*, imgdnn_err_code*) {
  return {};
}

imgdnn_err_code __attribute__((weak))
imgdnnNetworkBinaryDestroy(imgdnn_network_binary*) {
  return IMGDNN_SUCCESS;
}
