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
#include "backends/imgdnn/compilation.hpp"
#include "backends/imgdnn/convert.hpp"
#include "common/device.hpp"
#include "common/model.hpp"

#include <fstream>
#include <sstream>
#include <streambuf>

ResultCode ANeuralNetworksCompilation_create(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation) {
  ANeuralNetworksDevice* device = nullptr;
  TENSOROPT_RETURN_IF_ERROR(ANeuralNetworks_getDevice(0, &device));
  TENSOROPT_RETURN_IF_ERROR(ANeuralNetworksCompilation_createForDevices(
      model, &device, 1, compilation));
  (*compilation)->owned_device.reset(device, [](ANeuralNetworksDevice* device) {
    ANeuralNetworksDevice_free(device);
  });
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksCompilation_createForDevices(
    ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
    uint32_t num_devices, ANeuralNetworksCompilation** compilation) {
  TENSOROPT_RETURN_IF_NULL(model);
  TENSOROPT_RETURN_IF_UNFINISHED(model);

  if (num_devices != 1) {
    VLOG_AT("Error: Expected one device but got " << num_devices);
    return ANEURALNETWORKS_BAD_DATA;
  }

  *compilation = new ANeuralNetworksCompilation();
  (*compilation)->model = model;
  auto rt_device = devices[0];
  (*compilation)->device = rt_device;

  imgdnn_err_code ret;
  auto cl_device = rt_device->queue->get_device().get();
  BACKEND_CALL_RET((*compilation)->imgdnn_context_, imgdnnCLCreateContext,
                   rt_device->queue->get_context().get(), 1, &cl_device,
                   IMGDNN_CTX_FLAGS_NONE, &(*compilation)->imgdnn_device_,
                   &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);
  (*compilation)->imgdnn_flags_ = IMGDNN_NETWORK_OBJ_FLAG_NONE;
  (*compilation)->imgdnn_binary_ = {0, nullptr};
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksCompilation_setCaching(
    ANeuralNetworksCompilation* compilation, const char* cache_dir,
    const uint8_t* token) {
  TENSOROPT_RETURN_IF_FINISHED(compilation);
  TENSOROPT_RETURN_IF_NULL(cache_dir);
  TENSOROPT_RETURN_IF_NULL(token);
  std::string str_cache_dir(cache_dir);
  std::stringstream ss;
  ss << str_cache_dir;
  if (str_cache_dir.back() != '/') {
    ss << '/';
  }
  ss << token[0];
  for (unsigned i = 1; i < BYTE_SIZE_OF_CACHE_TOKEN; ++i) {
    ss << '_' << token[i];
  }
  compilation->token_path = ss.str();
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksCompilation_setPreference(ANeuralNetworksCompilation*,
                                                    int32_t) {
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksCompilation_finish(
    ANeuralNetworksCompilation* compilation) {
  if (compilation->finished) {
    return ANEURALNETWORKS_NO_ERROR;
  }

  if (!compilation->serialized) {
    TENSOROPT_RETURN_IF_ERROR(convertModel(compilation));
  }

  imgdnn_err_code ret;
  BACKEND_CALL_RET(compilation->imgdnn_network_object_,
                   imgdnnCreateNetworkObject, compilation->imgdnn_device_,
                   compilation->imgdnn_context_, compilation->imgdnn_network_,
                   static_cast<unsigned>(compilation->imgdnn_inputs_.size()),
                   compilation->imgdnn_inputs_.data(),
                   static_cast<unsigned>(compilation->imgdnn_outputs_.size()),
                   compilation->imgdnn_outputs_.data(),
                   compilation->imgdnn_flags_,
                   compilation->imgdnn_options_.c_str(), &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);

  compilation->finished = true;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksCompilation_serialize(
    ANeuralNetworksCompilation* compilation, void** data,
    std::size_t* data_size) {
  if (!compilation->token_path.empty()) {
    std::ifstream file(compilation->token_path,
                       std::ios::in | std::ios::binary | std::ios::ate);
    if (file.good()) {
      std::ostringstream buffer;
      buffer << file.rdbuf();
      const std::string& file_str(buffer.str());
      compilation->cached_file.assign(file_str.begin(), file_str.end());
      *data_size = compilation->cached_file.size();
      *data = compilation->cached_file.data();
      return ANEURALNETWORKS_NO_ERROR;
    }
  }

  if (!compilation->finished && !compilation->serialized) {
    TENSOROPT_RETURN_IF_ERROR(convertModel(compilation));
  }

  imgdnn_err_code ret;
  BACKEND_CALL_RET(compilation->imgdnn_binary_, imgdnnCreateNetworkBinary,
                   compilation->imgdnn_device_, compilation->imgdnn_context_,
                   compilation->imgdnn_network_,
                   static_cast<unsigned>(compilation->imgdnn_inputs_.size()),
                   compilation->imgdnn_inputs_.data(),
                   static_cast<unsigned>(compilation->imgdnn_outputs_.size()),
                   compilation->imgdnn_outputs_.data(),
                   compilation->imgdnn_flags_,
                   compilation->imgdnn_options_.c_str(), &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);

  *data = compilation->imgdnn_binary_.data;
  *data_size = compilation->imgdnn_binary_.size;
  compilation->serialized = true;
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation) {
  if (!compilation) {
    return;
  }
  if (compilation->imgdnn_binary_.data) {
    BACKEND_CALL(imgdnnNetworkBinaryDestroy, &compilation->imgdnn_binary_);
  }
  if (compilation->finished) {
    BACKEND_CALL(imgdnnNetworkObjectDestroy,
                 compilation->imgdnn_network_object_);
    BACKEND_CALL(imgdnnNetworkDestroy, compilation->imgdnn_network_);
  }
  BACKEND_CALL(imgdnnContextDestroy, compilation->imgdnn_context_);
  delete compilation;
}
