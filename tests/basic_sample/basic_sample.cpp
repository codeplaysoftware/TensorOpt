/**
 * Sample from https://developer.android.com/ndk/guides/neuralnetworks
 */

#include <SYCL/sycl.hpp>
#include <tensoropt/tensoropt.hpp>

// Generate data
#include <algorithm>
#include <fstream>
#include <random>
#include <string>

// C functions to open a file and mmap memory
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void generate_data(const std::string& filename, unsigned seed,
                          unsigned size_bytes) {
  std::ofstream out(filename, std::ios::binary);
  std::generate_n(std::ostreambuf_iterator<char>(out), size_bytes,
                  std::mt19937{seed});
}

int main() {
  generate_data("training_data", 42, 96);

  // Create a memory buffer from the file that contains the trained data.
  ANeuralNetworksMemory* mem1 = NULL;
  int fd = open("training_data", O_RDONLY);
  std::size_t file_size = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);
  ANeuralNetworksMemory_createFromFd(file_size, PROT_READ, fd, 0, &mem1);
  close(fd);

  ANeuralNetworksModel* model = NULL;
  ANeuralNetworksModel_create(&model);

  // In our example, all our tensors are matrices of dimension [3][4].
  ANeuralNetworksOperandType tensor3x4Type;
  tensor3x4Type.type = ANEURALNETWORKS_TENSOR_FLOAT32;
  tensor3x4Type.scale = 0.f;  // These fields are useful for quantized tensors.
  tensor3x4Type.zeroPoint =
      0;  // These fields are useful for quantized tensors.
  tensor3x4Type.dimensionCount = 2;
  uint32_t dims[2] = {3, 4};
  tensor3x4Type.dimensions = dims;

  // We also specify operands that are activation function specifiers.
  ANeuralNetworksOperandType activationType;
  activationType.type = ANEURALNETWORKS_INT32;
  activationType.scale = 0.f;
  activationType.zeroPoint = 0;
  activationType.dimensionCount = 0;
  activationType.dimensions = NULL;

  // Now we add the seven operands, in the same order defined in the diagram.
  ANeuralNetworksModel_addOperand(model, &tensor3x4Type);   // operand 0
  ANeuralNetworksModel_addOperand(model, &tensor3x4Type);   // operand 1
  ANeuralNetworksModel_addOperand(model, &activationType);  // operand 2
  ANeuralNetworksModel_addOperand(model, &tensor3x4Type);   // operand 3
  ANeuralNetworksModel_addOperand(model, &tensor3x4Type);   // operand 4
  ANeuralNetworksModel_addOperand(model, &activationType);  // operand 5
  ANeuralNetworksModel_addOperand(model, &tensor3x4Type);   // operand 6

  // In our example, operands 1 and 3 are constant tensors whose value was
  // established during the training process.
  // The formula for size calculation is dim0 * dim1 * elementSize.
  const int sizeOfTensor = 3 * 4 * sizeof(float);
  ANeuralNetworksModel_setOperandValueFromMemory(model, 1, mem1, 0,
                                                 sizeOfTensor);
  ANeuralNetworksModel_setOperandValueFromMemory(model, 3, mem1, sizeOfTensor,
                                                 sizeOfTensor);

  // We set the values of the activation operands, in our example operands 2
  // and 5.
  int32_t noneValue = ANEURALNETWORKS_FUSED_NONE;
  ANeuralNetworksModel_setOperandValue(model, 2, &noneValue, sizeof(noneValue));
  ANeuralNetworksModel_setOperandValue(model, 5, &noneValue, sizeof(noneValue));

  // We have two operations in our example.
  // The first consumes operands 1, 0, 2, and produces operand 4.
  uint32_t addInputIndexes[3] = {1, 0, 2};
  uint32_t addOutputIndexes[1] = {4};
  ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3,
                                    addInputIndexes, 1, addOutputIndexes);

  // The second consumes operands 3, 4, 5, and produces operand 6.
  uint32_t multInputIndexes[3] = {3, 4, 5};
  uint32_t multOutputIndexes[1] = {6};
  ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3,
                                    multInputIndexes, 1, multOutputIndexes);

  // Our model has one input (0) and one output (6).
  uint32_t modelInputIndexes[1] = {0};
  uint32_t modelOutputIndexes[1] = {6};
  ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, modelInputIndexes, 1,
                                                modelOutputIndexes);

  ANeuralNetworksModel_finish(model);

  // Compile the model.
  ANeuralNetworksCompilation* compilation;
  ANeuralNetworksCompilation_create(model, &compilation);

  // Ask to optimize for low power consumption.
  ANeuralNetworksCompilation_setPreference(compilation,
                                           ANEURALNETWORKS_PREFER_LOW_POWER);

  ANeuralNetworksCompilation_finish(compilation);

  // Run the compiled model against a set of inputs.
  ANeuralNetworksExecution* run1 = NULL;
  ANeuralNetworksExecution_create(compilation, &run1);

  // Set the single input to our sample model. Since it is small, we won’t use a
  // memory buffer.
  float myInput[3][4];
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      myInput[i][j] = static_cast<float>(i * j);
    }
  }
  ANeuralNetworksExecution_setInput(run1, 0, NULL, myInput, sizeof(myInput));

  // Set the output.
  float myOutput[3][4];
  ANeuralNetworksExecution_setOutput(run1, 0, NULL, myOutput, sizeof(myOutput));

  // Starts the work. The work proceeds asynchronously.
  ANeuralNetworksEvent* run1_end = NULL;
  ANeuralNetworksExecution_startCompute(run1, &run1_end);

  // For our example, we have no other work to do and will just wait for the
  // completion.
  ANeuralNetworksEvent_wait(run1_end);
  ANeuralNetworksEvent_free(run1_end);
  ANeuralNetworksExecution_free(run1);

  // Apply the compiled model to a different set of inputs.
  ANeuralNetworksExecution* run2;
  ANeuralNetworksExecution_create(compilation, &run2);
  ANeuralNetworksExecution_setInput(run2, 0, NULL, myInput, sizeof(myInput));
  ANeuralNetworksExecution_setOutput(run2, 0, NULL, myOutput, sizeof(myOutput));
  ANeuralNetworksEvent* run2_end = NULL;
  ANeuralNetworksExecution_startCompute(run2, &run2_end);
  ANeuralNetworksEvent_wait(run2_end);
  ANeuralNetworksEvent_free(run2_end);
  ANeuralNetworksExecution_free(run2);

  // Cleanup
  ANeuralNetworksCompilation_free(compilation);
  ANeuralNetworksModel_free(model);
  ANeuralNetworksMemory_free(mem1);

  return 0;
}
