// Copyright (C) Codeplay Software Limited.
#ifndef INCLUDE_TENSOROPT_OPERATION_HPP
#define INCLUDE_TENSOROPT_OPERATION_HPP

/**
 * Available operations.
 *
 * Binary operations will reshape and broadcast the inputs if needed.
 * The rank of the output is the maximum rank of the inputs.
 * Example:
 * lhs: [4, 1, 2]
 * rhs: [5, 4, 3, 1]
 * res: [4, 4, 3, 2]
 */
enum OperationCode : int {
  /**
   * Binary addition operation.
   *
   * Inputs:
   * 0: TENSOR_* tensor - lhs
   * 1: TENSOR_* tensor - rhs of the same type as lhs
   * 2: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as lhs
   */
  ANEURALNETWORKS_ADD,

  /**
   * Compute a forward 2D average pooling.
   *
   * Inputs:
   * 0: TENSOR_FLOAT32 tensor - input in NHWC format
   * 1: INT32 scalar - PaddingCode
   * 2: INT32 scalar - stride width
   * 3: INT32 scalar - stride height
   * 4: INT32 scalar - filter width
   * 5: INT32 scalar - filter height
   * 6: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   * 7: BOOL scalar  - Set to true to use NCHW format (optional, defaults to
   *                   false)
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_AVERAGE_POOL_2D,

  /**
   * Cast a tensor to a new type.
   *
   * Input:
   * 0: TENSOR_* tensor - tensor to cast
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same shape as the input
   */
  ANEURALNETWORKS_CAST,

  /**
   * Concatenate multiple tensors along a dimension.
   *
   * Inputs:
   * 0 to (n-1): TENSOR_* tensor - n inputs, of the same type and of shape
   *                               [D0, D1, ..., Daxis(i), ..., Dm]
   * n: INT32 scalar - concatenation axis, negative values specify dimensions
   *                   from the end
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as the inputs and of
   *                      shape [D0, D1, ..., sum(Daxis(i)), ..., Dm]
   */
  ANEURALNETWORKS_CONCATENATION,

  /**
   * Compute a forward 2D convolution.
   *
   * Inputs:
   * 0:  TENSOR_FLOAT32 tensor - input in NHWC format
   * 1:  TENSOR_FLOAT32 tensor - filter in [Co, Fh, Fw, Ci] format
   * 2:  TENSOR_FLOAT32 tensor - bias 1D tensor of size Co or 0D to ignore bias
   * 3:  INT32 scalar - PaddingCode
   * 4:  INT32 scalar - stride width
   * 5:  INT32 scalar - stride height
   * 6:  INT32 scalar - FuseCode (optional, defaults to
   *                    ANEURALNETWORKS_FUSED_NONE)
   * 7:  BOOL scalar  - Set to true to specify the input in NCHW format
   *                    (optional, defaults to false)
   * 8:  BOOL scalar  - Set to true to specify the filter in [Fh, Fw, Ci, Co]
   *                    format (optional, defaults to false)
   * 9:  INT32 scalar - dilation width (optional, defaults to 1)
   * 10: INT32 scalar - dilation height (optional, defaults to 1)
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_CONV_2D,

  /**
   * Binary division operation.
   *
   * Inputs:
   * 0: TENSOR_* tensor - lhs
   * 1: TENSOR_* tensor - rhs of the same type as lhs
   * 2: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as lhs
   */
  ANEURALNETWORKS_DIV,

  /**
   * Compute a 2D depthwise convolution.
   *
   * Inputs:
   * 0:  TENSOR_FLOAT32 tensor - input in NHWC format
   * 1:  TENSOR_FLOAT32 tensor - filter in [Co, Fh, Fw, Ci] format
   * 2:  TENSOR_FLOAT32 tensor - bias 1D tensor of size Co or 0D to ignore bias
   * 3:  INT32 scalar - PaddingCode
   * 4:  INT32 scalar - stride width
   * 5:  INT32 scalar - stride height
   * 6:  INT32 scalar - depthwise multiplier (optional, defaults to 1)
   * 7:  INT32 scalar - FuseCode (optional, defaults to
   *                    ANEURALNETWORKS_FUSED_NONE)
   * 8:  BOOL scalar  - Set to true to specify the input in NCHW format
   *                    (optional, defaults to false)
   * 9:  BOOL scalar  - Set to true to specify the filter in [Fh, Fw, Ci, Co]
   *                    format (optional, defaults to false)
   * 10: INT32 scalar - dilation width (optional, defaults to 1)
   * 11: INT32 scalar - dilation height (optional, defaults to 1)
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_DEPTHWISE_CONV_2D,

  /**
   * Unary exp operation
   *
   * Input:
   * 0: TENSOR_FLOAT32 tensor - input
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_EXP,

  /**
   * Binary maximum operation.
   *
   * Inputs:
   * 0: TENSOR_* tensor - lhs
   * 1: TENSOR_* tensor - rhs of the same type as lhs
   * 2: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as lhs
   */
  ANEURALNETWORKS_MAX,

  /**
   * Compute a forward 2D max pooling.
   *
   * Inputs:
   * 0: TENSOR_FLOAT32 tensor - input in NHWC format
   * 1: INT32 scalar - PaddingCode
   * 2: INT32 scalar - stride width
   * 3: INT32 scalar - stride height
   * 4: INT32 scalar - filter width
   * 5: INT32 scalar - filter height
   * 6: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   * 7: BOOL scalar  - Set to true to use NCHW format (optional, defaults to
   *                   false)
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_MAX_POOL_2D,

  /**
   * Binary minimum operation.
   *
   * Inputs:
   * 0: TENSOR_* tensor - lhs, must be of rank 2
   * 1: TENSOR_* tensor - rhs of the same type as lhs, must be of rank 2
   * 2: BOOL scalar - Whether to transpose lhs
   * 3: BOOL scalar - Whether to transpose rhs
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as lhs
   */
  ANEURALNETWORKS_MATMUL,

  /**
   * Binary minimum operation.
   *
   * Inputs:
   * 0: TENSOR_* tensor - lhs
   * 1: TENSOR_* tensor - rhs of the same type as lhs
   * 2: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as lhs
   */
  ANEURALNETWORKS_MIN,

  /**
   * Binary multiplication operation.
   *
   * Inputs:
   * 0: TENSOR_* tensor - lhs
   * 1: TENSOR_* tensor - rhs of the same type as lhs
   * 2: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as lhs
   */
  ANEURALNETWORKS_MUL,

  /**
   * Unary relu operation, compute max(0, input)
   *
   * Input:
   * 0: TENSOR_FLOAT32 tensor - input
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_RELU,

  /**
   * Unary relu1 operation, compute min(1, max(-1, input))
   *
   * Input:
   * 0: TENSOR_FLOAT32 tensor - input
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_RELU1,

  /**
   * Unary relu6 operation, compute min(6, max(0, input))
   *
   * Input:
   * 0: TENSOR_FLOAT32 tensor - input
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_RELU6,

  /**
   * Reshape a tensor.
   *
   * Inputs:
   * 0: TENSOR_* tensor - input
   * 1: INT32 vector - the numer of element in this shape must be the
   *                   same as in the input's shape. At most one component
   *                   can be -1, this component will be computed so that
   *                   the number of elements stays the same.
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as the input
   */
  ANEURALNETWORKS_RESHAPE,

  /**
   * Unary rsqrt operation
   *
   * Input:
   * 0: TENSOR_FLOAT32 tensor - input
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_RSQRT,

  /**
   * Extract slices from a tensor.
   *
   * Inputs:
   * 0: TENSOR_* tensor - input
   * 1: INT32 vector - begin of the slices, must be of size rank(input).
   * 2: INT32 vector - size of the slices, must be of size rank(input).
   *                   Use a negative value to select the whole size of a
   *                   specific dimension.
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as the input
   */
  ANEURALNETWORKS_SLICE,

  /**
   * Compute the softmax function of a tensor.
   * Equivalent of
   * output[i] = exp((input[i] - max(input, axis)) * beta) /
   *     sum_{j}(exp((input[j] - max(input, axis)) * beta), axis)
   *
   * Inputs:
   * 0: TENSOR_FLOAT32 tensor - input
   * 1: FLOAT32 scalar - beta factor (optional, defaults to 1)
   * 2: INT32 scalar - specify the axis to reduce, negative values specify
   *                   dimensions from the end (optional, defaults to -1)
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_SOFTMAX,

  /**
   * Unary sqrt operation
   *
   * Input:
   * 0: TENSOR_FLOAT32 tensor - input
   *
   * Output:
   * 0: TENSOR_FLOAT32 tensor - result of the same type as the input
   */
  ANEURALNETWORKS_SQRT,

  /**
   * Squeeze a tensor to remove all dimensions of size 1.
   *
   * Inputs:
   * 0: TENSOR_* tensor - input
   * 1: INT32 vector - set of dimensions allowed to be squeezed,
   *                   negative values specify dimensions from the end
   *                   (optional, defaults to all)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as the input
   */
  ANEURALNETWORKS_SQUEEZE,

  /**
   * Extract strided slices from a tensor.
   * The slice at the ith dimension is of size
   * ceil((end[i] - begin[i]) / stride[i]).
   * The stride starts at begin[i], is incremented by stride[i] and stops at
   * end[i] excluded. A negative stride is possible to reverse the stride.
   * Assuming positive begin and end, begin must be smaller than end for a
   * positive stride and bigger for a negative stride.
   * The masks can change this behaviour as explained.
   *
   * Inputs:
   * 0: TENSOR_* tensor - input
   * 1: INT32 vector - begin of the slices, must be of size smaller or equal
   *                   to rank(input)
   * 2: INT32 vector - end of the slices, must be of the same size than begin
   *                   input
   * 3: INT32 vector - strides of the slices, must be non-zero values of the
   *                   same size than begin input
   * 4: INT32 scalar - begin_mask, if the ith bit is set behave as if
   *                   begin[i]=0 (optional, defaults to 0)
   * 5: INT32 scalar - end_mask, if the ith bit is set behave as if end[i]=-1
   *                   (optional, defaults to 0)
   * 6: INT32 scalar - shrink_axis_mask, if the ith bit is set, shrink the ith
   *                   dimension to a single element specified by begin[i],
   *                   end[i] and strides[i] are ignored (optional, defaults
   *                   to 0)
   * 7: INT32 scalar - ellipsis_mask, only one bit can be set at maximum.
   *                   Inserts as many missing dimensions as needed and
   *                   select the whole slice for these dimensions.
   *                   Allows begin, end and strides to be smaller than
   *                   rank(input) (optional, defaults to 0)
   * 8: INT32 scalar - new_axis_mask, if the ith bit is set, reshape the result
   *                   to insert a dimension of size 1 at the ith position
   *                   (optional, defaults to 0)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as the input
   */
  ANEURALNETWORKS_STRIDED_SLICE,

  /**
   * Binary substraction operation.
   *
   * Inputs:
   * 0: TENSOR_* tensor - lhs
   * 1: TENSOR_* tensor - rhs of the same type as lhs
   * 2: INT32 scalar - FuseCode (optional, defaults to
   *                   ANEURALNETWORKS_FUSED_NONE)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as lhs
   */
  ANEURALNETWORKS_SUB,

  /**
   * Transpose, or shuffle the dimensions of a tensor.
   *
   * Inputs:
   * 0: TENSOR_* tensor - input
   * 1: INT32 vector - perm, describes the permutations to apply, must be of
   *                   size rank(input)
   *
   * Output:
   * 0: TENSOR_* tensor - result of the same type as the input
   */
  ANEURALNETWORKS_TRANSPOSE,

  /**
   * Not a valid operation.
   * Used to get the number of supported operations (see
   * ANeuralNetworksModel_getSupportedOperationsForQueue)
   */
  ANEURALNETWORKS_OPERATION_COUNT,
};

using ANeuralNetworksOperationType = OperationCode;

/**
 * Available fused activation functions.
 */
enum FuseCode : int {
  ANEURALNETWORKS_FUSED_NONE,
  ANEURALNETWORKS_FUSED_RELU,
  ANEURALNETWORKS_FUSED_RELU1,
  ANEURALNETWORKS_FUSED_RELU6,
};

/**
 * Available paddings.
 */
enum PaddingCode : int {
  ANEURALNETWORKS_PADDING_SAME,
  ANEURALNETWORKS_PADDING_VALID,
};

#endif  // INCLUDE_TENSOROPT_OPERATION_HPP
