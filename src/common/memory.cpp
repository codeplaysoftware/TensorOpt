// Copyright (C) Codeplay Software Limited.
#include "common/memory.hpp"
#include "common/macro.hpp"

#ifndef _WIN32
#include <sys/mman.h>
#endif

ResultCode ANeuralNetworksMemory_createFromFd(std::size_t size, int protect,
                                              int fd, std::size_t offset,
                                              ANeuralNetworksMemory** memory) {
  TENSOROPT_RETURN_IF_COND(fd < 0, "Error: invalid fid",
                           ANEURALNETWORKS_BAD_DATA);
#ifdef _WIN32
  VLOG_AT(
      "Unsupported function on Windows, use "
      "ANeuralNetworksMemory_createFromHost instead");
  return ANEURALNETWORKS_BAD_DATA;
#else
  void* data = mmap(nullptr, size, protect, MAP_PRIVATE | MAP_POPULATE, fd,
                    static_cast<unsigned>(offset));
  TENSOROPT_RETURN_IF_COND(data == nullptr, "Error: mmap failed",
                           ANEURALNETWORKS_BAD_DATA);
  TENSOROPT_RETURN_IF_ERROR(
      ANeuralNetworksMemory_createFromHost(data, size, memory));
  int ret = munmap(data, size);
  TENSOROPT_RETURN_IF_COND(ret != 0, "Error: munmap failed",
                           ANEURALNETWORKS_BAD_DATA);
  return ANEURALNETWORKS_NO_ERROR;
#endif
}

ResultCode ANeuralNetworksMemory_createFromHost(
    const void* data, std::size_t size, ANeuralNetworksMemory** memory) {
  TENSOROPT_RETURN_IF_NULL(data);
  TENSOROPT_RETURN_IF_NULL(memory);
  *memory = new ANeuralNetworksMemory{tensoropt_buffer_t(
      static_cast<const tensoropt_buffer_t::value_type*>(data),
      cl::sycl::range<1>(size))};
  (*memory)->buffer.set_final_data(nullptr);
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksMemory_createFromBuffer(
    const tensoropt_buffer_t& buffer, ANeuralNetworksMemory** memory) {
  TENSOROPT_RETURN_IF_NULL(memory);
  *memory = new ANeuralNetworksMemory{buffer};
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksMemory_resetBuffer(ANeuralNetworksMemory* memory,
                                             const tensoropt_buffer_t& buffer) {
  memory->buffer = buffer;
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksMemory_free(ANeuralNetworksMemory* memory) {
  if (memory) {
    delete memory;
  }
}
