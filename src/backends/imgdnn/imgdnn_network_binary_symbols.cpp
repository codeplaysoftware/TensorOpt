// Copyright (C) Codeplay Software Limited.
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
