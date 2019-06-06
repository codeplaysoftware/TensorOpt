// Copyright (C) Codeplay Software Limited.
#ifndef SRC_BACKENDS_IMGDNN_BACKEND_HPP
#define SRC_BACKENDS_IMGDNN_BACKEND_HPP

#include <iostream>
#include <tuple>

#include <imgdnn/cl.h>
#include <imgdnn/imgdnn.h>

#include "common/backend_print.hpp"
#include "common/macro.hpp"
#include "common/utils.hpp"

#define IMGDNN_PRINT_ERR(RET) VLOG_AT("Internal IMGDNN error: " << (RET))

#define IMGDNN_RETURN_IF_ERROR(RET) \
  if (RET != IMGDNN_SUCCESS) {      \
    IMGDNN_PRINT_ERR(RET);          \
    return;                         \
  }

#define IMGDNN_RETURN_ERR_IF_ERROR(RET) \
  if (RET != IMGDNN_SUCCESS) {          \
    IMGDNN_PRINT_ERR(RET);              \
    return ANEURALNETWORKS_INCOMPLETE;  \
  }

inline imgdnn_tensor_descriptor get_img_td(imgdnn_tensor img_tensor) {
  imgdnn_tensor_descriptor img_td;
  auto ret = imgdnnGetTensorDescriptor(img_tensor, &img_td);
  if (ret != IMGDNN_SUCCESS) {
    IMGDNN_PRINT_ERR(ret);
  }
  return img_td;
}

inline std::ostream& operator<<(std::ostream& os,
                                const imgdnn_network_binary& imgdnn_bin) {
  return os << "{data=" << imgdnn_bin.data << ", size=" << imgdnn_bin.size
            << "}";
}

inline std::ostream& operator<<(std::ostream& os,
                                const imgdnn_quant_param& imgdnn_quant_param) {
  return os << "{scale=" << imgdnn_quant_param.scale
            << ", zero_point=" << imgdnn_quant_param.zero_point << "}";
}

inline std::ostream& operator<<(std::ostream& os,
                                const imgdnn_tensor_descriptor& imgdnn_td) {
  return os << "{type=" << imgdnn_td.type
            << ", dimensions=" << imgdnn_td.dimensions << ", size=["
            << arrayToString(imgdnn_td.size, imgdnn_td.dimensions)
            << "], quant_param=" << imgdnn_td.quant_param << "}";
}

inline std::ostream& operator<<(std::ostream& os, imgdnn_tensor img_tensor) {
  auto img_td = get_img_td(img_tensor);
  // imgdnn_tensor is an opaque pointer, it is casted to avoid recursion
  return os << static_cast<const void*>(img_tensor) << " (" << img_td << ")";
}

inline std::ostream& operator<<(std::ostream& os,
                                const imgdnn_tensor_descriptor* imgdnn_td) {
  return printPointer(os, imgdnn_td);
}

inline std::ostream& operator<<(std::ostream& os, const imgdnn_err_code* ret) {
  return printPointer(os, ret);
}

/*
 * Overload printHostData to take an imgdnn_tensor_descriptor as a shape
 * descriptor.
 * data is a void* to avoid the caller doing any casting.
 */
template <class T>
inline void printHostData(const void* data,
                          const imgdnn_tensor_descriptor* const imgdnn_td) {
  imgdnn_err_code ret;
  auto data_count = imgdnnGetDescriptorSize(imgdnn_td, &ret) / sizeof(T);
  IMGDNN_RETURN_IF_ERROR(ret);
  printHostData(static_cast<const T*>(data), data_count);
}

template <class... Args>
struct BackendPrintFunc<decltype(&imgdnnNetworkTransposeOp), Args...> {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    TENSOROPT_UNUSED_VARIABLE(func_name);
    VLOG(func_name << "(");
    auto tuple_args = std::tuple<Args...>(args...);
    auto img_tensor = std::get<1>(tuple_args);
    auto img_td = get_img_td(img_tensor);
    auto order = std::get<2>(tuple_args);
    VLOG(std::get<0>(tuple_args) << ", ");
    VLOG(img_tensor << ", ");
    printHostData(order, img_td.dimensions);
    VLOG(", " << std::get<3>(tuple_args));
    VLOG(")");
  }
};

template <class... Args>
struct BackendPrintFunc<decltype(&imgdnnNetworkSubTensor), Args...> {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    TENSOROPT_UNUSED_VARIABLE(func_name);
    VLOG(func_name << "(");
    auto tuple_args = std::tuple<Args...>(args...);
    auto img_tensor = std::get<1>(tuple_args);
    auto img_td = get_img_td(img_tensor);
    auto start = std::get<2>(tuple_args);
    auto end = std::get<3>(tuple_args);
    auto stride = std::get<4>(tuple_args);
    VLOG(std::get<0>(tuple_args) << ", ");
    VLOG(img_tensor << ", ");
    printHostData(start, img_td.dimensions);
    VLOG(", ");
    printHostData(end, img_td.dimensions);
    VLOG(", ");
    printHostData(stride, img_td.dimensions);
    VLOG(", " << std::get<5>(tuple_args));
    VLOG(")");
  }
};

template <class... Args>
struct BackendPrintFunc<decltype(&imgdnnNetworkConcatOp), Args...> {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    TENSOROPT_UNUSED_VARIABLE(func_name);
    VLOG(func_name << "(");
    auto tuple_args = std::tuple<Args...>(args...);
    auto num_concats = std::get<3>(tuple_args);
    VLOG(std::get<0>(tuple_args) << ", ");
    printHostData(std::get<1>(tuple_args), num_concats);
    VLOG(", " << std::get<2>(tuple_args));
    VLOG(", " << num_concats);
    VLOG(", " << std::get<4>(tuple_args));
    VLOG(")");
  }
};

template <class... Args>
struct BackendPrintFunc<decltype(&imgdnnNetworkReduceOp), Args...> {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    TENSOROPT_UNUSED_VARIABLE(func_name);
    VLOG(func_name << "(");
    auto tuple_args = std::tuple<Args...>(args...);
    auto axis = std::get<3>(tuple_args);
    auto num_axis = std::get<4>(tuple_args);
    VLOG(std::get<0>(tuple_args) << ", ");
    VLOG(std::get<1>(tuple_args) << ", ");
    VLOG(std::get<2>(tuple_args) << ", ");
    printHostData(axis, num_axis);
    VLOG(", " << num_axis);
    VLOG(", " << std::get<5>(tuple_args));
    VLOG(")");
  }
};

template <class... Args>
struct BackendPrintFunc<decltype(&imgdnnNetworkFixedInput), Args...> {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    TENSOROPT_UNUSED_VARIABLE(func_name);
    VLOG(func_name << "(");
    auto tuple_args = std::tuple<Args...>(args...);
    auto imgdnn_td = std::get<1>(tuple_args);
    auto data = std::get<2>(tuple_args);
    VLOG(std::get<0>(tuple_args) << ", ");
    VLOG(imgdnn_td << ", ");
    switch (imgdnn_td->type) {
      case IMGDNN_TYPE_I8:
        printHostData<int8_t>(data, imgdnn_td);
        break;
      case IMGDNN_TYPE_U8:
        printHostData<uint8_t>(data, imgdnn_td);
        break;
      case IMGDNN_TYPE_I16:
        printHostData<int16_t>(data, imgdnn_td);
        break;
      case IMGDNN_TYPE_U16:
        printHostData<uint16_t>(data, imgdnn_td);
        break;
      case IMGDNN_TYPE_I32:
        printHostData<int32_t>(data, imgdnn_td);
        break;
      case IMGDNN_TYPE_U32:
        printHostData<uint32_t>(data, imgdnn_td);
        break;
      case IMGDNN_TYPE_F32:
        printHostData<float>(data, imgdnn_td);
        break;
      default:
        VLOG(data);
        break;
    }
    VLOG(", " << std::get<3>(tuple_args));
    VLOG(")");
  }
};

template <class... Args>
void printCreateNetworkHelper(const std::string& func_name, Args&&... args) {
  TENSOROPT_UNUSED_VARIABLE(func_name);
  VLOG(func_name << "(");
  auto tuple_args = std::tuple<Args...>(args...);
  auto num_inputs = std::get<3>(tuple_args);
  auto num_outputs = std::get<5>(tuple_args);
  VLOG(std::get<0>(tuple_args) << ", ");
  VLOG(std::get<1>(tuple_args) << ", ");
  VLOG(std::get<2>(tuple_args) << ", ");
  VLOG(num_inputs << ", ");
  printHostData(std::get<4>(tuple_args), num_inputs);
  VLOG(", " << num_outputs << ", ");
  printHostData(std::get<6>(tuple_args), num_outputs);
  VLOG(", " << std::get<7>(tuple_args) << ", ");
  VLOG("\"" << std::get<8>(tuple_args) << "\", ");
  VLOG(std::get<9>(tuple_args));
  VLOG(")");
}

template <class... Args>
struct BackendPrintFunc<decltype(&imgdnnCreateNetworkObject), Args...> {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    printCreateNetworkHelper(func_name, std::forward<Args>(args)...);
  }
};

template <class... Args>
struct BackendPrintFunc<decltype(&imgdnnCreateNetworkBinary), Args...> {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    printCreateNetworkHelper(func_name, std::forward<Args>(args)...);
  }
};

#endif  // SRC_BACKENDS_IMGDNN_BACKEND_HPP
