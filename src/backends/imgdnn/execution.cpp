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
#include "backends/imgdnn/execution.hpp"

#include <SYCL/codeplay.hpp>

#include "backends/imgdnn/compilation.hpp"
#include "common/device.hpp"
#include "common/event.hpp"
#include "common/memory.hpp"

static ResultCode createCommon(ANeuralNetworksExecution* execution) {
  imgdnn_err_code ret;
  BACKEND_CALL_RET(execution->imgdnn_binding_, imgdnnCreateBinding, &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);

  unsigned num_inputs;
  BACKEND_CALL_RET(ret, imgdnnNetworkObjectGetInputs,
                   execution->imgdnn_network_object_, 0, nullptr, &num_inputs);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);
  execution->imgdnn_inputs_.resize(num_inputs);
  BACKEND_CALL_RET(ret, imgdnnNetworkObjectGetInputs,
                   execution->imgdnn_network_object_, num_inputs,
                   execution->imgdnn_inputs_.data(), nullptr);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);

  unsigned num_outputs;
  BACKEND_CALL_RET(ret, imgdnnNetworkObjectGetOutputs,
                   execution->imgdnn_network_object_, 0, nullptr, &num_outputs);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);
  execution->imgdnn_outputs_.resize(num_outputs);
  BACKEND_CALL_RET(ret, imgdnnNetworkObjectGetOutputs,
                   execution->imgdnn_network_object_, num_outputs,
                   execution->imgdnn_outputs_.data(), nullptr);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);
  execution->identified_memory_lock = std::unique_lock<std::mutex>(
      execution->identified_memory_mutex, std::defer_lock);

  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_create(
    ANeuralNetworksCompilation* compilation,
    ANeuralNetworksExecution** execution) {
  TENSOROPT_RETURN_IF_NULL(compilation);
  TENSOROPT_RETURN_IF_UNFINISHED(compilation);
  *execution = new ANeuralNetworksExecution();
  (*execution)->created_from_compilation = true;
  (*execution)->device = compilation->device;
  (*execution)->imgdnn_network_object_ = compilation->imgdnn_network_object_;
  (*execution)->imgdnn_device_ = compilation->imgdnn_device_;
  (*execution)->imgdnn_context_ = compilation->imgdnn_context_;
  return createCommon(*execution);
}

ResultCode ANeuralNetworksExecution_createFromBinary(
    const void* data, std::size_t data_size,
    const ANeuralNetworksDevice* device, ANeuralNetworksExecution** execution) {
  TENSOROPT_RETURN_IF_NULL(data);
  *execution = new ANeuralNetworksExecution();
  (*execution)->device = device;

  auto cl_device = device->queue->get_device().get();
  imgdnn_err_code ret;
  BACKEND_CALL_RET((*execution)->imgdnn_context_, imgdnnCLCreateContext,
                   device->queue->get_context().get(), 1, &cl_device,
                   IMGDNN_CTX_FLAGS_NONE, &(*execution)->imgdnn_device_, &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);

  BACKEND_CALL_RET((*execution)->imgdnn_network_object_,
                   imgdnnLoadNetworkObject, (*execution)->imgdnn_device_,
                   (*execution)->imgdnn_context_, data_size, data, &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);

  return createCommon(*execution);
}

ResultCode ANeuralNetworksExecution_setInput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const void* data,
    std::size_t length) {
  TENSOROPT_UNUSED_VARIABLE(type);

  // Optional inputs are not added
  if (data && length > 0) {
    imgdnn_err_code ret;
    imgdnn_memory img_memory;
    BACKEND_CALL_RET(img_memory, imgdnnImportMemory, execution->imgdnn_context_,
                     const_cast<void*>(data), length,
                     IMGDNN_IMPORT_MEM_TYPE_CPU, &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    TENSOROPT_TO_UINT32_INDEX(index, uindex);
    BACKEND_CALL_RET(ret, imgdnnBindingAddInput, execution->imgdnn_binding_,
                     execution->imgdnn_inputs_[uindex], img_memory);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    // Store the memory objects to free them after the execution
    execution->imgdnn_memories_.push_back(img_memory);
  }
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_setInputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    std::size_t offset, std::size_t length) {
  TENSOROPT_UNUSED_VARIABLE(type);
  if (offset != 0) {
    // TODO: handle offset by adding subtensors in the compilation stage?
    VLOG_AT("Error: non-zero offsets are not supported");
    return ANEURALNETWORKS_BAD_DATA;
  }

  // Optional inputs are not added
  if (memory && length > 0) {
    // Memory object is const_casted here to be able to create accessors from
    // the underlying buffer
    auto cc_memory = const_cast<ANeuralNetworksMemory*>(memory);
    TENSOROPT_TO_UINT32_INDEX(index, uindex);
    std::lock_guard<std::mutex> lock(execution->identified_memory_mutex);
    execution->identified_memory_inputs[uindex] = {cc_memory, offset, length};
  }
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_setOutput(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, void* data, std::size_t length) {
  TENSOROPT_UNUSED_VARIABLE(type);

  // Optional outputs are not added
  if (data && length > 0) {
    imgdnn_err_code ret;
    imgdnn_memory img_memory;
    BACKEND_CALL_RET(img_memory, imgdnnImportMemory, execution->imgdnn_context_,
                     data, length, IMGDNN_IMPORT_MEM_TYPE_CPU, &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    TENSOROPT_TO_UINT32_INDEX(index, uindex);
    BACKEND_CALL_RET(ret, imgdnnBindingAddOutput, execution->imgdnn_binding_,
                     execution->imgdnn_outputs_[uindex], img_memory);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    // Store the memory objects to be able to lock them later
    execution->host_output_memories.emplace_back(data, img_memory);
  }
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_setOutputFromMemory(
    ANeuralNetworksExecution* execution, int32_t index,
    const ANeuralNetworksOperandType* type, const ANeuralNetworksMemory* memory,
    std::size_t offset, std::size_t length) {
  TENSOROPT_UNUSED_VARIABLE(type);
  if (offset != 0) {
    // TODO: handle offset by adding subtensors in the compilation stage?
    VLOG_AT("Error: non-zero offsets are not supported");
    return ANEURALNETWORKS_BAD_DATA;
  }

  // Optional outputs are not added
  if (memory && length > 0) {
    // Memory object is const_casted here to be able to create accessors from
    // the underlying buffer
    auto cc_memory = const_cast<ANeuralNetworksMemory*>(memory);
    TENSOROPT_TO_UINT32_INDEX(index, uindex);
    std::lock_guard<std::mutex> lock(execution->identified_memory_mutex);
    execution->identified_memory_outputs[uindex] = {cc_memory, offset, length};
  }
  return ANEURALNETWORKS_NO_ERROR;
}

static void imgdnnDescriptorToRTOperandType(
    ANeuralNetworksExecution* execution,
    const imgdnn_tensor_descriptor& descriptor,
    ANeuralNetworksOperandType& op) {
  switch (descriptor.type) {
    case IMGDNN_TYPE_I8:
    case IMGDNN_TYPE_U8:
      op.type = ANEURALNETWORKS_TENSOR_BOOL8;
      break;

    case IMGDNN_TYPE_I32:
    case IMGDNN_TYPE_U32:
      op.type = ANEURALNETWORKS_TENSOR_INT32;
      break;

    case IMGDNN_TYPE_F32:
      op.type = ANEURALNETWORKS_TENSOR_FLOAT32;
      break;

    default:
      op.type = ANEURALNETWORKS_INVALID;
  }
  op.dimensionCount = descriptor.dimensions;
  execution->dimensions.emplace_back();
  auto& topt_dims = execution->dimensions.back();
  topt_dims.assign(descriptor.size, descriptor.size + descriptor.dimensions);
  op.dimensions = topt_dims.data();
}

uint32_t ANeuralNetworksExecution_getIdentifiedInputCount(
    const ANeuralNetworksExecution* execution) {
  return static_cast<uint32_t>(execution->imgdnn_inputs_.size());
}

ResultCode ANeuralNetworksExecution_getIdentifiedInputs(
    ANeuralNetworksExecution* execution, ANeuralNetworksOperandType* inputs) {
  imgdnn_err_code ret;
  for (std::size_t i = 0; i < execution->imgdnn_inputs_.size(); ++i) {
    imgdnn_tensor_descriptor descriptor;
    BACKEND_CALL_RET(descriptor, imgdnnGetInputDescriptor,
                     execution->imgdnn_inputs_[i], &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    imgdnnDescriptorToRTOperandType(execution, descriptor, inputs[i]);
  }
  return ANEURALNETWORKS_NO_ERROR;
}

uint32_t ANeuralNetworksExecution_getIdentifiedOutputCount(
    const ANeuralNetworksExecution* execution) {
  return static_cast<uint32_t>(execution->imgdnn_outputs_.size());
}

ResultCode ANeuralNetworksExecution_getIdentifiedOutputs(
    ANeuralNetworksExecution* execution, ANeuralNetworksOperandType* outputs) {
  imgdnn_err_code ret;
  for (std::size_t i = 0; i < execution->imgdnn_outputs_.size(); ++i) {
    imgdnn_tensor_descriptor descriptor;
    BACKEND_CALL_RET(descriptor, imgdnnGetOutputDescriptor,
                     execution->imgdnn_outputs_[i], &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    imgdnnDescriptorToRTOperandType(execution, descriptor, outputs[i]);
  }
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_getOutputOperandDimensions(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* dimensions) {
  imgdnn_err_code ret;
  imgdnn_tensor_descriptor descriptor;
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  BACKEND_CALL_RET(descriptor, imgdnnGetOutputDescriptor,
                   execution->imgdnn_outputs_[uindex], &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);
  for (std::size_t i = 0; i < descriptor.dimensions; ++i) {
    dimensions[i] = static_cast<uint32_t>(descriptor.size[i]);
  }
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_getOutputOperandRank(
    ANeuralNetworksExecution* execution, int32_t index, uint32_t* rank) {
  imgdnn_err_code ret;
  imgdnn_tensor_descriptor descriptor;
  TENSOROPT_TO_UINT32_INDEX(index, uindex);
  BACKEND_CALL_RET(descriptor, imgdnnGetOutputDescriptor,
                   execution->imgdnn_outputs_[uindex], &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);
  *rank = descriptor.dimensions;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_compute(
    ANeuralNetworksExecution* execution) {
  ANeuralNetworksEvent* event;
  TENSOROPT_RETURN_IF_ERROR(
      ANeuralNetworksExecution_startCompute(execution, &event));
  TENSOROPT_RETURN_IF_ERROR(ANeuralNetworksEvent_wait(event));
  ANeuralNetworksEvent_free(event);
  return ANEURALNETWORKS_NO_ERROR;
}

/**
 * @brief Print an error if a function call to the backend failed
 * In an interop_task we only print the error to avoid any synchronisation
 * @param ret error code
 */
inline void interopCheckImgdnnErr(imgdnn_err_code ret) {
  TENSOROPT_UNUSED_VARIABLE(ret);
#ifdef VERBOSE_LOG
  if (ret != IMGDNN_SUCCESS) {
    VLOG_AT("Error: IMGDNN execution failed with code " << ret);
  }
#endif
}

/**
 * @brief Fetches IMGDNN memory from an accessor
 * @tparam AccT Accessor type
 * @return imgdnn_memory object
 */
template <class AccT>
inline imgdnn_memory importImgMemory(
    ANeuralNetworksExecution* execution, const AccT& acc,
    const cl::sycl::codeplay::interop_handle& h) {
  imgdnn_memory img_memory;
  const auto length = acc.get_size();
  imgdnn_err_code ret;
  BACKEND_CALL_RET(img_memory, imgdnnImportMemory, execution->imgdnn_context_,
                   h.get(acc), length, IMGDNN_IMPORT_MEM_TYPE_OPENCL, &ret);
  interopCheckImgdnnErr(ret);
  return img_memory;
}

ResultCode ANeuralNetworksExecution_startCompute(
    ANeuralNetworksExecution* execution, ANeuralNetworksEvent** output_event) {
  // identified_memory_lock will be unlocked during the interop_task.
  // The lock is stored inside the execution to keep it alive long enough.
  execution->identified_memory_lock.lock();
  auto& queue = execution->device->queue;
  auto sycl_event = queue->submit([execution](
                                      cl::sycl::codeplay::handler& cgh) {
    execution->input_indexed_accessors.clear();
    for (const auto& input_pair : execution->identified_memory_inputs) {
      execution->input_indexed_accessors.emplace_back(
          input_pair.first, input_pair.second.memory->buffer
                                .get_access<cl::sycl::access::mode::read>(cgh));
    }
    execution->output_indexed_accessors.clear();
    for (const auto& output_pair : execution->identified_memory_outputs) {
      execution->output_indexed_accessors.emplace_back(
          output_pair.first,
          output_pair.second.memory->buffer
              .get_access<cl::sycl::access::mode::write>(cgh));
    }
    cgh.interop_task([execution](const cl::sycl::codeplay::interop_handle& h) {
      // imgdnn_memories objects need to be copied locally so that the next
      // execution is not blocked
      std::vector<imgdnn_memory> task_memories;
      task_memories.reserve(execution->imgdnn_memories_.size() +
                            execution->input_indexed_accessors.size() +
                            execution->output_indexed_accessors.size());
      task_memories = execution->imgdnn_memories_;
      execution->imgdnn_memories_.clear();
      imgdnn_err_code ret;
      // Bind inputs
      for (const auto& acc_pair : execution->input_indexed_accessors) {
        imgdnn_memory img_memory =
            importImgMemory(execution, acc_pair.second, h);
        BACKEND_CALL_RET(ret, imgdnnBindingAddInput, execution->imgdnn_binding_,
                         execution->imgdnn_inputs_[acc_pair.first], img_memory);
        interopCheckImgdnnErr(ret);
        task_memories.push_back(img_memory);
      }

      // Bind outputs
      for (const auto& acc_pair : execution->output_indexed_accessors) {
        imgdnn_memory img_memory =
            importImgMemory(execution, acc_pair.second, h);
        BACKEND_CALL_RET(
            ret, imgdnnBindingAddOutput, execution->imgdnn_binding_,
            execution->imgdnn_outputs_[acc_pair.first], img_memory);
        interopCheckImgdnnErr(ret);
        task_memories.push_back(img_memory);
      }
      execution->identified_memory_lock.unlock();

      // The IMGDNN execution is made blocking so that the returned
      // SYCL event represents the execution of the whole graph.
      BACKEND_CALL_RET(ret, imgdnnNetworkObjectExecute,
                       execution->imgdnn_network_object_,
                       execution->imgdnn_binding_, true, 0, nullptr, nullptr);
      interopCheckImgdnnErr(ret);

      for (auto img_mem : task_memories) {
        BACKEND_CALL_RET(ret, imgdnnMemoryDestroy, img_mem);
        interopCheckImgdnnErr(ret);
      }
    });
  });
  execution->dimensions.clear();

  if (output_event) {
    *output_event = new ANeuralNetworksEvent(sycl_event, execution);
  }

  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksExecution_notifyWait(
    ANeuralNetworksExecution* execution) {
  imgdnn_err_code ret;
  // Lock the memory to copy data from device to host
  void* output_ptr = nullptr;
  for (const auto& hom : execution->host_output_memories) {
    BACKEND_CALL_RET(output_ptr, imgdnnMemoryLock, hom.img_mem,
                     IMGDNN_LOCK_ACCESS_READ_ONLY, &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    TENSOROPT_RETURN_IF_COND(
        hom.data != output_ptr,
        "Error: IMGDNN returned a different host pointer from imported memory",
        ANEURALNETWORKS_BAD_DATA);
    BACKEND_CALL_RET(ret, imgdnnMemoryUnlock, hom.img_mem);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);

    // Destroy output memory now it's been read
    BACKEND_CALL_RET(ret, imgdnnMemoryDestroy, hom.img_mem);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
  }
  execution->host_output_memories.clear();
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksExecution_free(ANeuralNetworksExecution* execution) {
  if (!execution) {
    return;
  }
  // If compilation was provided it will free its own imgdnn object
  if (!execution->created_from_compilation) {
    BACKEND_CALL(imgdnnNetworkObjectDestroy, execution->imgdnn_network_object_);
    BACKEND_CALL(imgdnnContextDestroy, execution->imgdnn_context_);
  }
  BACKEND_CALL(imgdnnBindingDestroy, execution->imgdnn_binding_);
  delete execution;
}
