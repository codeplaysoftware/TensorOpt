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
#include "backends/imgdnn/convert.hpp"

#include <bitset>
#include <cstring>

#include "backends/imgdnn/compilation.hpp"
#include "common/device.hpp"
#include "common/model.hpp"
#include "common/utils.hpp"

namespace {

struct Converter {
 private:
  // Create negative indices used for internal imgdnn_tensors with a specific
  // meaning.
  enum SpecialImgTensor : int64_t {
    // Create a unique imgdnn_tensor with constant value one
    CONST_FLOAT32_ONE = 1
  };

  ANeuralNetworksCompilation* compilation;  // weak_ptr
  const ANeuralNetworksModel* model;        // weak_ptr

  // Store all the imgdnn_tensor created during the conversions
  // Indices can be stricly negative for internal tensors or positive for
  // tensors mapping to an operand
  using indexed_img_tensors = std::unordered_map<int64_t, imgdnn_tensor>;
  indexed_img_tensors img_tensors;

  imgdnn_err_code ret;

  // Offset to convert an exclusive end bound to an inclusive one
  static constexpr int INCLUSIVE_END = -1;

 public:
  Converter(ANeuralNetworksCompilation* c)
      : compilation(c), model(c->model), img_tensors() {}

  Converter(const Converter&) = delete;
  Converter(Converter&&) = default;
  Converter& operator=(const Converter&) = delete;
  Converter& operator=(Converter&&) = default;

  ResultCode operator()() {
    // Convert const device operands to const host operands owned.
    // This is needed because IMGDNN backend does not support providing a const
    // device operand to the network. This makes sense when serializing the
    // model since all the constant data has to be on the host. This is an
    // unecessary overhead if the network is not serialized as we would move the
    // data from device to host here and IMGDNN will move it back to the device
    // when executing. TensorFlow does not use
    // ANeuralNetworksModel_setOperandValueFromMemory so this is not an
    // important issue.
    std::vector<cl::sycl::event> copy_events;
    for (auto& pair : model->const_device_operands) {
      auto& const_host_operand =
          compilation->const_copied_to_host_operands[pair.first];
      // We remove the const here as the model must stay constant but submitting
      // the copy requires a non-const reference to the buffer. The buffer could
      // be copied but we would then need to wait after each submit.
      auto& const_device_operand =
          const_cast<ANeuralNetworksModel::ConstDeviceOperand&>(pair.second);
      const_host_operand.resize(const_device_operand.length);
      auto host_ptr = const_host_operand.data();
      tensoropt_buffer_t& buffer = const_device_operand.memory.buffer;
      auto event =
          compilation->device->queue->submit([&](cl::sycl::handler& cgh) {
            auto acc = buffer.get_access<cl::sycl::access::mode::read>(
                cgh, cl::sycl::range<1>(const_device_operand.length),
                cl::sycl::id<1>(const_device_operand.offset));
            cgh.copy(acc, host_ptr);
          });
      copy_events.push_back(event);
    }

    for (auto& event : copy_events) {
      event.wait_and_throw();
    }

    // Add network inputs
    for (uint32_t op_idx : model->inputs) {
      imgdnn_tensor_descriptor img_td;
      const auto& op = model->operands[op_idx];
      TENSOROPT_RETURN_IF_ERROR(RTOperandTypeToImg(op, img_td));
      imgdnn_tensor img_tensor;
      BACKEND_CALL_RET(img_tensor, imgdnnNetworkInput,
                       compilation->imgdnn_network_, &img_td, &ret);
      IMGDNN_RETURN_ERR_IF_ERROR(ret);
      if (!img_tensors.insert({op_idx, img_tensor}).second) {
        VLOG_AT("Error: Input index " << op_idx
                                      << " was identified multiple times");
        return ANEURALNETWORKS_BAD_DATA;
      }
      compilation->imgdnn_inputs_.push_back(img_tensor);
    }

    // Add network operations
    // NNAPI doesn't assume any order in which the operations must be
    // added but IMGDNN requires that they are added in execution order (by
    // construction).
    // TensorFlow already adds the operations in the correct order
    // TODO: Sort the operations with a pre order traversal
    for (std::size_t op_idx = 0; op_idx < model->operations.size(); ++op_idx) {
      const auto& operation = model->operations[op_idx];
      switch (operation.type) {
        case ANEURALNETWORKS_EXP:
        case ANEURALNETWORKS_RELU:
        case ANEURALNETWORKS_RELU1:
        case ANEURALNETWORKS_RELU6:
        case ANEURALNETWORKS_RSQRT:
        case ANEURALNETWORKS_SQRT:
          TENSOROPT_RETURN_IF_ERROR(convertUnary(operation));
          break;

        case ANEURALNETWORKS_ADD:
        case ANEURALNETWORKS_MUL:
        case ANEURALNETWORKS_SUB:
        case ANEURALNETWORKS_DIV:
        case ANEURALNETWORKS_MAX:
        case ANEURALNETWORKS_MIN:
          TENSOROPT_RETURN_IF_ERROR(convertBinary(operation));
          break;

        case ANEURALNETWORKS_AVERAGE_POOL_2D:
        case ANEURALNETWORKS_MAX_POOL_2D:
          TENSOROPT_RETURN_IF_ERROR(convertPool(operation));
          break;

        case ANEURALNETWORKS_CONV_2D:
        case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
          TENSOROPT_RETURN_IF_ERROR(convertConv2D(operation));
          break;

        case ANEURALNETWORKS_MATMUL:
          TENSOROPT_RETURN_IF_ERROR(convertMatmul(operation));
          break;

        case ANEURALNETWORKS_TRANSPOSE:
          TENSOROPT_RETURN_IF_ERROR(convertTranspose(operation));
          break;

        case ANEURALNETWORKS_RESHAPE:
          TENSOROPT_RETURN_IF_ERROR(convertReshape(operation));
          break;

        case ANEURALNETWORKS_SQUEEZE:
          TENSOROPT_RETURN_IF_ERROR(convertSqueeze(operation));
          break;

        case ANEURALNETWORKS_CONCATENATION:
          TENSOROPT_RETURN_IF_ERROR(convertConcat(operation));
          break;

        case ANEURALNETWORKS_SLICE:
          TENSOROPT_RETURN_IF_ERROR(convertSlice(operation));
          break;

        case ANEURALNETWORKS_STRIDED_SLICE:
          TENSOROPT_RETURN_IF_ERROR(convertStridedSlice(operation));
          break;

        case ANEURALNETWORKS_SOFTMAX:
          TENSOROPT_RETURN_IF_ERROR(convertSoftmax(operation));
          break;

        case ANEURALNETWORKS_CAST:
          TENSOROPT_RETURN_IF_ERROR(convertCast(operation));
          break;

        default:
          VLOG_AT("Unsupported operation " << operation.type
                                           << " at operation index " << op_idx);
          return ANEURALNETWORKS_OP_FAILED;
      }
      // Check the IMGDNN output size matches what TensorOpt expects
      for (unsigned i = 0; i < operation.outputs.size(); ++i) {
        auto output_idx = operation.outputs[i];
        const ANeuralNetworksOperandType& output_op =
            model->operands[output_idx];
        imgdnn_tensor_descriptor img_td;
        BACKEND_CALL_RET(ret, imgdnnGetTensorDescriptor,
                         img_tensors[output_idx], &img_td);
        if (!areShapesEqual(output_op, img_td)) {
          VLOG_AT(
              "Unexpected output shape when converting operation #"
              << op_idx << " (code=" << operation.type << "), output #" << i
              << ": IMGDNN returned ["
              << arrayToString(img_td.size, img_td.dimensions)
              << "] but TensorOpt expected ["
              << arrayToString(output_op.dimensions, output_op.dimensionCount)
              << "]");
          return ANEURALNETWORKS_OP_FAILED;
        }
      }
    }

    // Add network outputs
    for (auto output_idx : model->outputs) {
      compilation->imgdnn_outputs_.push_back(img_tensors[output_idx]);
    }

    return ANEURALNETWORKS_NO_ERROR;
  }

 private:
  /**
   * Return true if the TensorOpt and IMGDNN shapes are equal.
   * The 0D TensorOpt shape and 1D IMGDNN shape of size 1 are considered to
   * be equal. An IMGDNN shape cannot be 0D.
   */
  bool areShapesEqual(const ANeuralNetworksOperandType& topt_op,
                      const imgdnn_tensor_descriptor& img_td) {
    if (topt_op.dimensionCount == 0 && img_td.dimensions == 1 &&
        img_td.size[0] == 1) {
      return true;
    }
    bool are_shapes_equal = topt_op.dimensionCount == img_td.dimensions;
    unsigned dim = 0;
    while (are_shapes_equal && dim < topt_op.dimensionCount) {
      are_shapes_equal = topt_op.dimensions[dim] == img_td.size[dim];
      ++dim;
    }
    return are_shapes_equal;
  }

  /**
   * Convert RT enum types to IMGDNN types.
   */
  ResultCode RTCodeToImg(ANeuralNetworksOperandCode code, imgdnn_type& type) {
    switch (code) {
      case ANEURALNETWORKS_FLOAT32:
      case ANEURALNETWORKS_TENSOR_FLOAT32:
        type = IMGDNN_TYPE_F32;
        return ANEURALNETWORKS_NO_ERROR;

      case ANEURALNETWORKS_INT32:
      case ANEURALNETWORKS_TENSOR_INT32:
        type = IMGDNN_TYPE_I32;
        return ANEURALNETWORKS_NO_ERROR;

      case ANEURALNETWORKS_UINT32:
        type = IMGDNN_TYPE_U32;
        return ANEURALNETWORKS_NO_ERROR;

      case ANEURALNETWORKS_BOOL:
      case ANEURALNETWORKS_TENSOR_BOOL8:
        type = IMGDNN_TYPE_U8;
        return ANEURALNETWORKS_NO_ERROR;

      default:
        VLOG_AT("Internal error: invalid OperandCode " << code);
        return ANEURALNETWORKS_BAD_DATA;
    }
  }

  /**
   * Convert RT operand type to IMGDNN tensor descriptor.
   */
  ResultCode RTOperandTypeToImg(const ANeuralNetworksOperandType& op,
                                imgdnn_tensor_descriptor& img_td) {
    TENSOROPT_RETURN_IF_ERROR(RTCodeToImg(op.type, img_td.type));
    if (op.dimensionCount == 0) {
      img_td.dimensions = 1;
      img_td.size[0] = 1;
    } else {
      img_td.dimensions = op.dimensionCount;
      for (unsigned i = 0; i < op.dimensionCount; ++i) {
        img_td.size[i] = op.dimensions[i];
      }
    }
    img_td.quant_param.scale = 0.f;
    img_td.quant_param.zero_point = 0;
    return ANEURALNETWORKS_NO_ERROR;
  }

  /**
   * Try to read a const operand at index idx from a given map.
   * Return whether a const operand exists at idx.
   */
  bool readConstHostOperandFromMap(
      uint32_t idx, const void** data, std::size_t& length,
      const ANeuralNetworksCompilation::owned_const_host_operands& operands) {
    auto it = operands.find(idx);
    if (it != operands.end()) {
      *data = it->second.data();
      length = it->second.size();
      return true;
    }
    return false;
  }

  /**
   * Try to read a const operand at index idx.
   * Return whether a const operand exists at idx.
   */
  bool readConstHostOperandHelper(uint32_t idx, const void** data,
                                  std::size_t& length) {
    if (readConstHostOperandFromMap(
            idx, data, length, compilation->const_copied_to_host_operands)) {
      return true;
    }
    if (readConstHostOperandFromMap(idx, data, length,
                                    model->const_host_operands_owned)) {
      return true;
    }
    auto it = model->const_host_operands.find(idx);
    if (it != model->const_host_operands.end()) {
      *data = it->second.data;
      length = it->second.length;
      return true;
    }
    return false;
  }

  /**
   * Read a const host operand (owned or not) at index idx.
   */
  ResultCode readConstHostOperand(uint32_t idx, const void** data,
                                  std::size_t& length) {
    if (!readConstHostOperandHelper(idx, data, length)) {
      VLOG_AT("Error: Provided index " << idx
                                       << " was not added as an operand.");
      return ANEURALNETWORKS_BAD_DATA;
    }
    return ANEURALNETWORKS_NO_ERROR;
  }

  /**
   * Read a scalar host constant at the given index.
   */
  template <class T>
  ResultCode readConstHostOperand(uint32_t idx, T& value) {
    const void* data = nullptr;
    std::size_t length;
    TENSOROPT_RETURN_IF_ERROR(readConstHostOperand(idx, &data, length));
    if (length != sizeof(T)) {
      VLOG_AT("Error: Operand at index " << idx << " is of size " << length
                                         << " but expected " << sizeof(T));
      return ANEURALNETWORKS_BAD_DATA;
    }
    std::memcpy(&value, data, sizeof(T));
    return ANEURALNETWORKS_NO_ERROR;
  }

  /**
   * Read a vector host constant at the given index.
   */
  template <class T>
  ResultCode readConstHostOperand(uint32_t idx, std::vector<T>& values) {
    const void* data = nullptr;
    std::size_t length;
    TENSOROPT_RETURN_IF_ERROR(readConstHostOperand(idx, &data, length));
    auto typed_data = static_cast<const T*>(data);
    values.assign(typed_data, typed_data + (length / sizeof(T)));
    return ANEURALNETWORKS_NO_ERROR;
  }

  /**
   * Read a bitset host constant at the given index.
   */
  ResultCode readConstHostOperand(uint32_t idx, std::bitset<32>& values) {
    int32_t value;
    TENSOROPT_RETURN_IF_ERROR(readConstHostOperand(idx, value));
    values = std::bitset<32>(static_cast<unsigned long long>(value));
    return ANEURALNETWORKS_NO_ERROR;
  }

  /**
   * Read a scalar or vector host constant if idx was provided in the
   * operation's input. value is unchanged and ANEURALNETWORKS_NO_ERROR is
   * returned if idx was not provided.
   */
  template <class T>
  ResultCode readOptionalConstHostOperand(
      const ANeuralNetworksModel::Operation& operation, uint32_t idx,
      T& value) {
    if (idx < operation.inputs.size()) {
      TENSOROPT_RETURN_IF_ERROR(
          readConstHostOperand(operation.inputs[idx], value));
    }
    return ANEURALNETWORKS_NO_ERROR;
  }

  /**
   * Try to create a fixed input IMGDNN tensor.
   * If op_idx is set as a model constant input, img_tensor is created and added
   * to img_tensors, added is set to true. Otherwise added is set to false.
   */
  ResultCode addFixedInputTensor(uint32_t op_idx, imgdnn_tensor& img_tensor,
                                 bool& added) {
    const void* data = nullptr;
    std::size_t length;
    if (readConstHostOperandHelper(op_idx, &data, length)) {
      if (std::find(model->inputs.begin(), model->inputs.end(), op_idx) !=
          model->inputs.end()) {
        VLOG_AT("Error: Operand at index "
                << op_idx
                << " cannot be both a constant model operand and an input");
        return ANEURALNETWORKS_BAD_DATA;
      }
      if (std::find(model->outputs.begin(), model->outputs.end(), op_idx) !=
          model->outputs.end()) {
        VLOG_AT("Error: Operand at index "
                << op_idx
                << " cannot be both a constant model operand and an output");
        return ANEURALNETWORKS_BAD_DATA;
      }
      imgdnn_tensor_descriptor img_td;
      const auto& op = model->operands[op_idx];
      TENSOROPT_RETURN_IF_ERROR(RTOperandTypeToImg(op, img_td));
      uint32_t op_size = getOperandTypeSizeBytes(op);
      if (op_size != length) {
        VLOG_AT("Error: Operand at index "
                << op_idx << " was described with a total size of " << op_size
                << "B but set with a value of size " << length << "B");
        return ANEURALNETWORKS_BAD_DATA;
      }
      BACKEND_CALL_RET(img_tensor, imgdnnNetworkFixedInput,
                       compilation->imgdnn_network_, &img_td, data, &ret);
      IMGDNN_RETURN_ERR_IF_ERROR(ret);
      img_tensors[op_idx] = img_tensor;
      added = true;
    } else {
      added = false;
    }
    return ANEURALNETWORKS_NO_ERROR;
  }

  /**
   * Get an existing imgdnn_tensor or create one if it is a user input.
   */
  ResultCode getImgTensor(int64_t idx, imgdnn_tensor& img_tensor) {
    auto it = img_tensors.find(idx);
    if (it != img_tensors.end()) {
      img_tensor = it->second;
      return ANEURALNETWORKS_NO_ERROR;
    }
    if (idx >= 0) {
      auto op_idx = static_cast<uint32_t>(idx);
      bool added;
      TENSOROPT_RETURN_IF_ERROR(addFixedInputTensor(op_idx, img_tensor, added));
      if (added) {
        return ANEURALNETWORKS_NO_ERROR;
      }
    }
    // idx is either an internal tensor or is an output of an operation
    // that was not converted yet.
    VLOG_AT("Error: Tensor for operand index " << idx
                                               << " was not created yet.");
    return ANEURALNETWORKS_OP_FAILED;
  }

  /**
   * Get an existing internal imgdnn_tensor or create one if it doesn't exist
   * yet.
   */
  ResultCode getInternalImgTensor(SpecialImgTensor idx, imgdnn_tensor& img_t) {
    auto map_idx = -idx;
    auto it = img_tensors.find(map_idx);
    if (it != img_tensors.end()) {
      img_t = it->second;
      return ANEURALNETWORKS_NO_ERROR;
    }

    if (idx == CONST_FLOAT32_ONE) {
      // FLOAT_ONE must live at least as long as the compilation object
      static constexpr float FLOAT_ONE = 1;
      imgdnn_tensor_descriptor img_td;
      img_td.dimensions = 1;
      img_td.size[0] = 1;
      img_td.type = IMGDNN_TYPE_F32;
      BACKEND_CALL_RET(img_t, imgdnnNetworkFixedInput,
                       compilation->imgdnn_network_, &img_td, &FLOAT_ONE, &ret);
      IMGDNN_RETURN_ERR_IF_ERROR(ret);
      img_tensors[map_idx] = img_t;
      return ANEURALNETWORKS_NO_ERROR;
    }

    VLOG_AT("Internal error: Could not create internal tensor with index "
            << idx);
    return ANEURALNETWORKS_OP_FAILED;
  }

  template <class Container>
  ResultCode convertTransposeHelper(imgdnn_tensor img_in,
                                    const Container& order,
                                    imgdnn_tensor& img_out) {
    BACKEND_CALL_RET(img_out, imgdnnNetworkTransposeOp,
                     compilation->imgdnn_network_, img_in, order.data(), &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode getImgNCHWTensor(const ANeuralNetworksModel::Operation& operation,
                              uint32_t idx, bool is_input_nchw,
                              imgdnn_tensor& img_nchw_out) {
    imgdnn_tensor img_in;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[idx], img_in));
    if (is_input_nchw) {
      img_nchw_out = img_in;
      return ANEURALNETWORKS_NO_ERROR;
    }
    static constexpr std::array<int, 4> nhwc_to_nchw{0, 3, 1, 2};
    TENSOROPT_RETURN_IF_ERROR(
        convertTransposeHelper(img_in, nhwc_to_nchw, img_nchw_out));
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode getImgOIHWTensor(const ANeuralNetworksModel::Operation& operation,
                              uint32_t idx, bool is_input_hwio,
                              imgdnn_tensor& img_hwio_out) {
    imgdnn_tensor img_in;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[idx], img_in));
    if (is_input_hwio) {
      static constexpr std::array<int, 4> hwio_to_oihw{3, 2, 0, 1};
      TENSOROPT_RETURN_IF_ERROR(
          convertTransposeHelper(img_in, hwio_to_oihw, img_hwio_out));
    } else {  // ohwi to oihw
      static constexpr std::array<int, 4> ohwi_to_oihw{0, 3, 1, 2};
      TENSOROPT_RETURN_IF_ERROR(
          convertTransposeHelper(img_in, ohwi_to_oihw, img_hwio_out));
    }
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode getSameFormatImgTensor(bool is_input_nchw, imgdnn_tensor img_in,
                                    imgdnn_tensor& img_out) {
    if (is_input_nchw) {
      img_out = img_in;
      return ANEURALNETWORKS_NO_ERROR;
    }
    static constexpr std::array<int, 4> nchw_to_nhwc{0, 2, 3, 1};
    TENSOROPT_RETURN_IF_ERROR(
        convertTransposeHelper(img_in, nchw_to_nhwc, img_out));
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertBinaryHelper(OperationCode op_code, imgdnn_tensor img_in0,
                                 imgdnn_tensor img_in1,
                                 imgdnn_tensor& img_out) {
    // int is used instead of OperationCode here since the type needs to be
    // hashable
    static std::unordered_map<int, imgdnn_operation_binary> rt_to_img_op_code{
        {ANEURALNETWORKS_ADD, IMGDNN_OPERATION_ADD},
        {ANEURALNETWORKS_MUL, IMGDNN_OPERATION_MUL},
        {ANEURALNETWORKS_SUB, IMGDNN_OPERATION_SUB},
        {ANEURALNETWORKS_DIV, IMGDNN_OPERATION_DIV},
        {ANEURALNETWORKS_MAX, IMGDNN_OPERATION_MAX},
        {ANEURALNETWORKS_MIN, IMGDNN_OPERATION_MIN},
        {ANEURALNETWORKS_MATMUL, IMGDNN_OPERATION_MATMUL}};

    // Imgdnn will automatically reshape and broadcast tensors if needed
    BACKEND_CALL_RET(img_out, imgdnnNetworkBinaryOp,
                     compilation->imgdnn_network_, img_in0, img_in1,
                     rt_to_img_op_code.at(op_code), &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertUnaryHelper(OperationCode op_code, imgdnn_tensor img_in,
                                imgdnn_tensor& img_out) {
    // Special cases for operations that do not translate to a imgdnn unary op.
    if (op_code == ANEURALNETWORKS_RELU1) {
      BACKEND_CALL_RET(img_out, imgdnnNetworkReLUOp,
                       compilation->imgdnn_network_, img_in, true, -1.f, true,
                       1.f, 1.f, &ret);
      IMGDNN_RETURN_ERR_IF_ERROR(ret);
      return ANEURALNETWORKS_NO_ERROR;
    }
    if (op_code == ANEURALNETWORKS_RELU6) {
      BACKEND_CALL_RET(img_out, imgdnnNetworkReLUOp,
                       compilation->imgdnn_network_, img_in, true, 0.f, true,
                       6.f, 1.f, &ret);
      IMGDNN_RETURN_ERR_IF_ERROR(ret);
      return ANEURALNETWORKS_NO_ERROR;
    }
    if (op_code == ANEURALNETWORKS_RSQRT) {
      // Write "rsqrt(x)" as "1 / sqrt(x)"
      imgdnn_tensor sqrt_tensor;
      TENSOROPT_RETURN_IF_ERROR(
          convertUnaryHelper(ANEURALNETWORKS_SQRT, img_in, sqrt_tensor));
      imgdnn_tensor img_cst_one;
      TENSOROPT_RETURN_IF_ERROR(
          getInternalImgTensor(CONST_FLOAT32_ONE, img_cst_one));
      TENSOROPT_RETURN_IF_ERROR(convertBinaryHelper(
          ANEURALNETWORKS_DIV, img_cst_one, sqrt_tensor, img_out));
      return ANEURALNETWORKS_NO_ERROR;
    }

    // int is used instead of OperationCode here since the type needs to be
    // hashable
    static std::unordered_map<int, imgdnn_operation_unary> rt_to_img_op_code{
        {ANEURALNETWORKS_RELU, IMGDNN_OPERATION_RELU},
        {ANEURALNETWORKS_EXP, IMGDNN_OPERATION_EXP},
        {ANEURALNETWORKS_SQRT, IMGDNN_OPERATION_SQRT}};

    BACKEND_CALL_RET(img_out, imgdnnNetworkUnaryOp,
                     compilation->imgdnn_network_, img_in,
                     rt_to_img_op_code.at(op_code), &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode addOptionalFuseCode(int32_t fuse_code, imgdnn_tensor& img_out) {
    if (fuse_code == ANEURALNETWORKS_FUSED_NONE) {
      return ANEURALNETWORKS_NO_ERROR;
    }

    static std::unordered_map<int32_t, OperationCode> fuse_code_to_op_code{
        {ANEURALNETWORKS_FUSED_RELU, ANEURALNETWORKS_RELU},
        {ANEURALNETWORKS_FUSED_RELU1, ANEURALNETWORKS_RELU1},
        {ANEURALNETWORKS_FUSED_RELU6, ANEURALNETWORKS_RELU6}};

    TENSOROPT_RETURN_IF_ERROR(convertUnaryHelper(
        fuse_code_to_op_code.at(fuse_code), img_out, img_out));
    return ANEURALNETWORKS_NO_ERROR;
  }

  template <unsigned TrueIdx, unsigned FalseIdx>
  ResultCode getHWHelper(const ANeuralNetworksModel::Operation& operation,
                         uint32_t idx, bool format, int32_t& res_h,
                         int32_t& res_w) {
    auto input_idx = operation.inputs[idx];
    const auto& op = model->operands[input_idx];
    TENSOROPT_RETURN_IF_COND(op.dimensionCount != 4,
                             "Internal error: expected operand "
                                 << input_idx
                                 << " to have 4 dimensions but got "
                                 << op.dimensionCount,
                             ANEURALNETWORKS_OP_FAILED);
    if (format) {
      res_h = op.dimensions[TrueIdx];
      res_w = op.dimensions[TrueIdx + 1];
    } else {
      res_h = op.dimensions[FalseIdx];
      res_w = op.dimensions[FalseIdx + 1];
    }
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode getInputHW(const ANeuralNetworksModel::Operation& operation,
                        uint32_t idx, bool nchw_format, int32_t& in_h,
                        int32_t& in_w) {
    return getHWHelper<2, 1>(operation, idx, nchw_format, in_h, in_w);
  }

  ResultCode getFilterHW(const ANeuralNetworksModel::Operation& operation,
                         uint32_t idx, bool hwio_format, int32_t& filter_h,
                         int32_t& filter_w) {
    return getHWHelper<0, 1>(operation, idx, hwio_format, filter_h, filter_w);
  }

  ResultCode computePadding(int32_t padding_code, int32_t input, int32_t stride,
                            int32_t filter, int32_t dilation,
                            unsigned& img_pad_begin, unsigned& img_pad_end) {
    if (padding_code == ANEURALNETWORKS_PADDING_VALID) {
      img_pad_begin = 0;
      img_pad_end = 0;
    } else if (padding_code == ANEURALNETWORKS_PADDING_SAME) {
      int32_t effective_filter = (filter - 1) * dilation + 1;
      int32_t pad_needed =
          std::max(0, (roundRatioUp(input, stride) - 1) * stride +
                          effective_filter - input);
      int32_t pad_begin = pad_needed / 2;
      img_pad_begin = static_cast<unsigned>(pad_begin);
      img_pad_end = static_cast<unsigned>(pad_needed - pad_begin);
    } else {
      VLOG_AT("Internal error: unknown padding " << padding_code);
      return ANEURALNETWORKS_OP_FAILED;
    }
    return ANEURALNETWORKS_NO_ERROR;
  }

  template <class Container>
  ResultCode checkVectorEqualRank(uint32_t rank, const Container& container,
                                  const std::string& input_name) {
    TENSOROPT_UNUSED_VARIABLE(input_name);
    TENSOROPT_RETURN_IF_COND(
        container.size() != rank,
        "Error: '" << input_name << "' argument has " << container.size()
                   << " elements but input rank is " << rank << ".",
        ANEURALNETWORKS_OP_FAILED);
    return ANEURALNETWORKS_NO_ERROR;
  }

  template <class Container>
  ResultCode checkVectorSmallerOrEqualRank(uint32_t rank,
                                           const Container& container,
                                           const std::string& input_name) {
    TENSOROPT_UNUSED_VARIABLE(input_name);
    TENSOROPT_RETURN_IF_COND(
        container.size() > rank,
        "Error: '" << input_name << "' argument has " << container.size()
                   << " elements but input rank is " << rank << ".",
        ANEURALNETWORKS_OP_FAILED);
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertReshapeHelper(imgdnn_tensor img_in, uint32_t shape_op_idx,
                                  imgdnn_tensor& img_out) {
    imgdnn_tensor_descriptor img_td;
    TENSOROPT_RETURN_IF_ERROR(
        RTOperandTypeToImg(model->operands[shape_op_idx], img_td));
    BACKEND_CALL_RET(img_out, imgdnnNetworkReshapeOp,
                     compilation->imgdnn_network_, img_in, &img_td, &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertUnary(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, inputs, 1);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    TENSOROPT_RETURN_IF_ERROR(
        convertUnaryHelper(operation.type, img_in, img_out));

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertBinary(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(operation, inputs, 2, 3);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in0;
    imgdnn_tensor img_in1;
    int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in0));
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[1], img_in1));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 2, fuse_code));

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    TENSOROPT_RETURN_IF_ERROR(
        convertBinaryHelper(operation.type, img_in0, img_in1, img_out));
    TENSOROPT_RETURN_IF_ERROR(addOptionalFuseCode(fuse_code, img_out));

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertPool(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(operation, inputs, 6, 8);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    // int is used instead of OperationCode here since the type needs to be
    // hashable
    static std::unordered_map<int, imgdnn_pooling_type> rt_to_img_op_code{
        {ANEURALNETWORKS_AVERAGE_POOL_2D, IMGDNN_POOLING_AVERAGE},
        {ANEURALNETWORKS_MAX_POOL_2D, IMGDNN_POOLING_MAX}};

    imgdnn_tensor img_nchw_in;
    int32_t padding_code;
    int32_t stride_w;
    int32_t stride_h;
    int32_t filter_w;
    int32_t filter_h;
    int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
    bool is_input_nchw = false;
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[1], padding_code));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[2], stride_w));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[3], stride_h));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[4], filter_w));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[5], filter_h));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 6, fuse_code));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 7, is_input_nchw));
    TENSOROPT_RETURN_IF_ERROR(
        getImgNCHWTensor(operation, 0, is_input_nchw, img_nchw_in));

    int32_t in_h;
    int32_t in_w;
    TENSOROPT_RETURN_IF_ERROR(
        getInputHW(operation, 0, is_input_nchw, in_h, in_w));

    unsigned img_window[2] = {static_cast<unsigned>(filter_h),
                              static_cast<unsigned>(filter_w)};
    unsigned img_strides[2] = {static_cast<unsigned>(stride_h),
                               static_cast<unsigned>(stride_w)};
    unsigned img_pad_begin[2];
    unsigned img_pad_end[2];
    static constexpr int32_t POOLING_DILATION = 1;
    TENSOROPT_RETURN_IF_ERROR(computePadding(padding_code, in_h, stride_h,
                                             filter_h, POOLING_DILATION,
                                             img_pad_begin[0], img_pad_end[0]));
    TENSOROPT_RETURN_IF_ERROR(computePadding(padding_code, in_w, stride_w,
                                             filter_w, POOLING_DILATION,
                                             img_pad_begin[1], img_pad_end[1]));
    imgdnn_tensor img_nchw_out;
    BACKEND_CALL_RET(img_nchw_out, imgdnnNetworkPooling2dOp_v2,
                     compilation->imgdnn_network_, img_nchw_in, img_window,
                     img_strides, img_pad_begin, img_pad_end,
                     rt_to_img_op_code.at(operation.type), &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    TENSOROPT_RETURN_IF_ERROR(
        getSameFormatImgTensor(is_input_nchw, img_nchw_out, img_out));
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertConv2D(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(operation, inputs, 6, 11);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_nchw_in;
    imgdnn_tensor img_oihw_filter;
    int32_t padding_code;
    int32_t stride_w;
    int32_t stride_h;
    int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
    bool is_input_nchw = false;
    bool is_filter_hwio = false;
    int32_t dilation_w;
    int32_t dilation_h;
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[3], padding_code));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[4], stride_w));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[5], stride_h));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 6, fuse_code));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 7, is_input_nchw));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 8, is_filter_hwio));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 9, dilation_w));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 10, dilation_h));
    TENSOROPT_RETURN_IF_ERROR(
        getImgNCHWTensor(operation, 0, is_input_nchw, img_nchw_in));
    TENSOROPT_RETURN_IF_ERROR(
        getImgOIHWTensor(operation, 1, is_filter_hwio, img_oihw_filter));

    int32_t in_h;
    int32_t in_w;
    int32_t filter_h;
    int32_t filter_w;
    TENSOROPT_RETURN_IF_ERROR(
        getInputHW(operation, 0, is_input_nchw, in_h, in_w));
    TENSOROPT_RETURN_IF_ERROR(
        getFilterHW(operation, 1, is_filter_hwio, filter_h, filter_w));

    unsigned img_strides[2] = {static_cast<unsigned>(stride_h),
                               static_cast<unsigned>(stride_w)};
    unsigned img_dilations[2] = {static_cast<unsigned>(dilation_h),
                                 static_cast<unsigned>(dilation_w)};
    unsigned img_pad_begin[2];
    unsigned img_pad_end[2];
    TENSOROPT_RETURN_IF_ERROR(computePadding(padding_code, in_h, stride_h,
                                             filter_h, dilation_h,
                                             img_pad_begin[0], img_pad_end[0]));
    TENSOROPT_RETURN_IF_ERROR(computePadding(padding_code, in_w, stride_w,
                                             filter_w, dilation_w,
                                             img_pad_begin[1], img_pad_end[1]));
    imgdnn_tensor img_nchw_out;
    switch (operation.type) {
      case ANEURALNETWORKS_CONV_2D:
        BACKEND_CALL_RET(img_nchw_out, imgdnnNetworkConvolution2dOp_v2,
                         compilation->imgdnn_network_, img_nchw_in,
                         img_oihw_filter, img_strides, img_pad_begin,
                         img_pad_end, img_dilations, &ret);
        break;

      case ANEURALNETWORKS_DEPTHWISE_CONV_2D:
        BACKEND_CALL_RET(img_nchw_out, imgdnnNetworkDepthConvolution2dOp_v2,
                         compilation->imgdnn_network_, img_nchw_in,
                         img_oihw_filter, img_strides, img_pad_begin,
                         img_pad_end, img_dilations, &ret);
        break;

      default:
        VLOG_AT("Internal error: unexpected operation " << operation.type);
        return ANEURALNETWORKS_OP_FAILED;
    }
    IMGDNN_RETURN_ERR_IF_ERROR(ret);
    imgdnn_tensor img_same_input_format_out;
    TENSOROPT_RETURN_IF_ERROR(getSameFormatImgTensor(
        is_input_nchw, img_nchw_out, img_same_input_format_out));

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    auto bias_op_idx = operation.inputs[2];
    const auto& bias_op = model->operands[bias_op_idx];
    if (bias_op.dimensionCount == 0) {
      img_out = img_same_input_format_out;
    } else if (bias_op.dimensionCount == 1) {
      imgdnn_tensor img_bias;
      TENSOROPT_RETURN_IF_ERROR(getImgTensor(bias_op_idx, img_bias));
      TENSOROPT_RETURN_IF_ERROR(convertBinaryHelper(
          ANEURALNETWORKS_ADD, img_same_input_format_out, img_bias, img_out));
    } else {
      VLOG_AT("Error: Expected 0 or 1 dimensionCount for bias operand but got "
              << bias_op.dimensionCount);
      return ANEURALNETWORKS_OP_FAILED;
    }
    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertMatmul(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, inputs, 4);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in0;
    imgdnn_tensor img_in1;
    bool lhs_t = false;
    bool rhs_t = false;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in0));
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[1], img_in1));
    TENSOROPT_RETURN_IF_ERROR(readConstHostOperand(operation.inputs[2], lhs_t));
    TENSOROPT_RETURN_IF_ERROR(readConstHostOperand(operation.inputs[3], rhs_t));

    static constexpr std::array<int, 2> transpose_order{1, 0};
    if (lhs_t) {
      TENSOROPT_RETURN_IF_ERROR(
          convertTransposeHelper(img_in0, transpose_order, img_in0));
    }
    if (rhs_t) {
      TENSOROPT_RETURN_IF_ERROR(
          convertTransposeHelper(img_in1, transpose_order, img_in1));
    }

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    TENSOROPT_RETURN_IF_ERROR(
        convertBinaryHelper(operation.type, img_in0, img_in1, img_out));

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertTranspose(
      const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, inputs, 2);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    std::vector<int32_t> permutations;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[1], permutations));
    auto rank = model->operands[operation.inputs[0]].dimensionCount;
    TENSOROPT_RETURN_IF_ERROR(
        checkVectorEqualRank(rank, permutations, "permutations"));

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    TENSOROPT_RETURN_IF_ERROR(
        convertTransposeHelper(img_in, permutations, img_out));

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertReshape(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, inputs, 2);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));
    // No need to read the new shape argument, TensorOpt assumes it matches the
    // one provided as the output.

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    TENSOROPT_RETURN_IF_ERROR(
        convertReshapeHelper(img_in, operation.outputs[0], img_out));

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertSqueeze(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(operation, inputs, 1, 2);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));
    // No need to read axis

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    TENSOROPT_RETURN_IF_ERROR(
        convertReshapeHelper(img_in, operation.outputs[0], img_out));

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertConcat(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_MIN_SIZE(operation, inputs, 2);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    auto nb_tensors = static_cast<unsigned>(operation.inputs.size() - 1);
    std::vector<imgdnn_tensor> img_ins;
    int32_t axis;
    for (unsigned i = 0; i < nb_tensors; ++i) {
      img_ins.emplace_back();
      TENSOROPT_RETURN_IF_ERROR(
          getImgTensor(operation.inputs[i], img_ins.back()));
    }
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[nb_tensors], axis));
    if (axis < 0) {
      axis += model->operands[operation.inputs[0]].dimensionCount;
    }

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    BACKEND_CALL_RET(img_out, imgdnnNetworkConcatOp,
                     compilation->imgdnn_network_, img_ins.data(),
                     static_cast<unsigned>(axis), nb_tensors, &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertSlice(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, inputs, 3);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    std::vector<int32_t> begins;
    std::vector<int32_t> sizes;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[1], begins));
    TENSOROPT_RETURN_IF_ERROR(readConstHostOperand(operation.inputs[2], sizes));

    auto input_op = model->operands[operation.inputs[0]];
    auto rank = input_op.dimensionCount;
    TENSOROPT_RETURN_IF_ERROR(checkVectorEqualRank(rank, begins, "begins"));
    TENSOROPT_RETURN_IF_ERROR(checkVectorEqualRank(rank, sizes, "sizes"));
    std::vector<std::size_t> img_starts(rank);
    std::vector<std::size_t> img_ends(rank);
    std::vector<std::size_t> img_strides(rank, 1);
    for (unsigned i = 0; i < rank; ++i) {
      img_starts[i] = static_cast<std::size_t>(begins[i]);
      if (sizes[i] < 0) {
        img_ends[i] = static_cast<std::size_t>(
            static_cast<long>(input_op.dimensions[i]) + INCLUSIVE_END);
      } else {  // sizes[i] cannot be 0
        img_ends[i] =
            img_starts[i] + static_cast<std::size_t>(sizes[i] + INCLUSIVE_END);
      }
    }

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    BACKEND_CALL_RET(img_out, imgdnnNetworkSubTensor,
                     compilation->imgdnn_network_, img_in, img_starts.data(),
                     img_ends.data(), img_strides.data(), &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertStridedSlice(
      const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(operation, inputs, 4, 9);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    std::vector<int32_t> begins;
    std::vector<int32_t> ends;
    std::vector<int32_t> strides;
    std::bitset<32> begin_mask;
    std::bitset<32> end_mask;
    std::bitset<32> shrink_axis_mask;
    std::bitset<32> ellipsis_mask;
    std::bitset<32> new_axis_mask;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[1], begins));
    TENSOROPT_RETURN_IF_ERROR(readConstHostOperand(operation.inputs[2], ends));
    TENSOROPT_RETURN_IF_ERROR(
        readConstHostOperand(operation.inputs[3], strides));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 4, begin_mask));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 5, end_mask));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 6, shrink_axis_mask));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 7, ellipsis_mask));
    TENSOROPT_RETURN_IF_ERROR(
        readOptionalConstHostOperand(operation, 8, new_axis_mask));

    auto input_op = model->operands[operation.inputs[0]];
    auto rank = input_op.dimensionCount;
    TENSOROPT_RETURN_IF_ERROR(
        checkVectorSmallerOrEqualRank(rank, begins, "begins"));
    TENSOROPT_RETURN_IF_COND(ends.size() != begins.size(),
                             "Error: 'ends' argument is of size "
                                 << ends.size() << " but expected "
                                 << begins.size() << ".",
                             ANEURALNETWORKS_OP_FAILED);
    TENSOROPT_RETURN_IF_COND(strides.size() != begins.size(),
                             "Error: 'strides' argument is of size "
                                 << strides.size() << " but expected "
                                 << begins.size() << ".",
                             ANEURALNETWORKS_OP_FAILED);
    if (ellipsis_mask.none()) {
      TENSOROPT_RETURN_IF_ERROR(checkVectorEqualRank(rank, begins, "begins"));
    } else if (begins.size() < rank) {
      unsigned ellipsis = 0;
      while (!ellipsis_mask[ellipsis] && ellipsis < rank) {
        ++ellipsis;
      }
      if (ellipsis < rank) {
        std::size_t diff = rank - begins.size();
        begins.insert(begins.begin() + ellipsis, diff, 0);
        ends.insert(ends.begin() + ellipsis, diff, -1);
        strides.insert(strides.begin() + ellipsis, diff, 1);
      }
    }
    std::vector<std::size_t> img_starts(rank);
    std::vector<std::size_t> img_ends(rank);
    std::vector<std::size_t> img_strides(rank);
    for (unsigned i = 0; i < rank; ++i) {
      if (begin_mask[i] || begins[i] < 0) {
        img_starts[i] = 0;
      } else {
        img_starts[i] = static_cast<std::size_t>(begins[i]);
      }
      if (shrink_axis_mask[i]) {
        img_ends[i] = img_starts[i] + (1 + INCLUSIVE_END);
        img_strides[i] = 1;
        continue;
      }
      if (end_mask[i] || ends[i] < 0) {
        img_ends[i] = static_cast<std::size_t>(
            static_cast<long>(input_op.dimensions[i]) + INCLUSIVE_END);
      } else {  // ends[i] cannot be 0
        img_ends[i] = static_cast<std::size_t>(ends[i] + INCLUSIVE_END);
      }
      if (strides[i] <= 0) {
        VLOG_AT("Error: strides must be stricly positive but got ["
                << arrayToString(strides, rank) << "].");
        return ANEURALNETWORKS_OP_FAILED;
      }
      img_strides[i] = static_cast<std::size_t>(strides[i]);
    }

    imgdnn_tensor img_strided_slice;
    BACKEND_CALL_RET(img_strided_slice, imgdnnNetworkSubTensor,
                     compilation->imgdnn_network_, img_in, img_starts.data(),
                     img_ends.data(), img_strides.data(), &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    if (new_axis_mask.none()) {
      img_out = img_strided_slice;
    } else {
      // TensorOpt assumes the shape provided by the output is correct.
      TENSOROPT_RETURN_IF_ERROR(convertReshapeHelper(
          img_strided_slice, operation.outputs[0], img_out));
    }

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertSoftmax(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(operation, inputs, 1, 3);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    float beta = 1.f;
    int32_t axis = -1;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));
    TENSOROPT_RETURN_IF_ERROR(readOptionalConstHostOperand(operation, 1, beta));
    TENSOROPT_RETURN_IF_ERROR(readOptionalConstHostOperand(operation, 2, axis));

    if (axis < 0) {
      axis += model->operands[operation.inputs[0]].dimensionCount;
    }

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    BACKEND_CALL_RET(img_out, imgdnnNetworkSoftmaxOp,
                     compilation->imgdnn_network_, img_in, beta,
                     static_cast<unsigned>(axis), &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);

    return ANEURALNETWORKS_NO_ERROR;
  }

  ResultCode convertCast(const ANeuralNetworksModel::Operation& operation) {
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, inputs, 1);
    TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(operation, outputs, 1);

    imgdnn_tensor img_in;
    TENSOROPT_RETURN_IF_ERROR(getImgTensor(operation.inputs[0], img_in));
    auto output_op = model->operands[operation.outputs[0]];
    imgdnn_type img_dst_type;
    TENSOROPT_RETURN_IF_ERROR(RTCodeToImg(output_op.type, img_dst_type));
    imgdnn_quant_param img_dst_quant;
    img_dst_quant.scale = output_op.scale;
    img_dst_quant.zero_point = output_op.zeroPoint;

    imgdnn_tensor& img_out = img_tensors[operation.outputs[0]];
    BACKEND_CALL_RET(img_out, imgdnnNetworkCastOp, compilation->imgdnn_network_,
                     img_in, img_dst_type, &img_dst_quant, &ret);
    IMGDNN_RETURN_ERR_IF_ERROR(ret);

    return ANEURALNETWORKS_NO_ERROR;
  }
};

}  // end namespace

ResultCode convertModel(ANeuralNetworksCompilation* compilation) {
  imgdnn_err_code ret;
  BACKEND_CALL_RET(compilation->imgdnn_network_, imgdnnCreateNetwork, &ret);
  IMGDNN_RETURN_ERR_IF_ERROR(ret);

  Converter converter(compilation);
  return converter();
}
