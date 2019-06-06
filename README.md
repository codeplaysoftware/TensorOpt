# TensorOpt

TensorOpt is designed as a wrapper around ML graph libraries.
Its purpose is to be integrated in ML frameworks using SYCL such as TensorFlow.

## TensorOpt API

TensorOpt API is based on the [Android NNAPI](https://developer.android.com/ndk/reference/group/neural-networks.html) with some minor changes:

* Features related to quantized and half types are not supported yet:
  * OperandCode:
    * `ANEURALNETWORKS_TENSOR_FLOAT16`
    * `ANEURALNETWORKS_TENSOR_QUANT8_ASYMM`
    * `ANEURALNETWORKS_TENSOR_QUANT8_SYMM`
    * `ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL`
    * `ANEURALNETWORKS_TENSOR_QUANT16_SYMM`
    * `ANEURALNETWORKS_TENSOR_QUANT16_ASYMM`
  * Struct `ANeuralNetworksSymmPerChannelQuantParams`
  * Functions:
    * `ANeuralNetworksModel_setOperandSymmPerChannelQuantParams`
    * `ANeauralNetworksModel_relaxComputationFloat32toFloat16`
* Features related to "Duration" are not supported yet:
  * Enum `DurationCode`
  * Functions:
    * `ANeuralNetworksExecution_getDuration`
    * `ANeuralNetworksExecution_setMeasureTiming`
* Features related to "Burst execution" are not supported yet:
  * Struct `ANeuralNetworksBurst`
  * Functions:
    * `ANeuralNetworksBurst_create`
    * `ANeuralNetworksBurst_free`
    * `ANeuralNetworksExecution_burstCompute`
* Features related to "Hardware buffer" are not supported:
  * Function `ANeuralNetworksMemory_createFromAHardwareBuffer`
* Not all the operations are supported; some have additional optional parameters, see [operation.hpp](include/tensoropt/operation.hpp).
* Added `ANeauralNetworksModel_canAddOperation`, similar to `ANeauralNetworksModel_getSupportedOperationsForDevices` but takes into account previously added operations.
* Added `ANeauralNetworksCompilation_serialize` and `ANeauralNetworksExecution_createFromBinary` to serialize and deserialize a compiled model.
* Added various getter functions and optional parameters to facilitate the integration in TensorFlow.
* Added functions specific to SYCL to be able to use existing SYCL queues, buffers and events.

## Building TensorOpt

### Installing the dependencies
Install the dependencies with:
```bash
apt install build-essentials cmake git
```

### Selecting a backend
TensorOpt is always built for one specific backend selected at compile-time. Currently only IMGDNN is supported, add `-TENSOROPT_BACKEND=IMGDNN -DIMGDNN_DIR=path/to/imgdnn` to the CMake options.

### Building a shared library
By default TensorOpt will be built as a static library. A shared library can be built instead by adding `-DBUILD_SHARED_LIBS=ON` to the CMake options.

### Cross-compiling for ARM
To cross-compile on ARM, the instructions are a subset of what is needed for cross-compiling TensorFlow:
1. Download the ARM Linaro toolchain
  ```
  # Set GCC_LIBARO_PATH to any new folder
  GCC_LINARO_PATH=path/to/linaro_toolchain
  cd ${GCC_LINARO_PATH}
  # Download and extract the toolchain
  wget https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
  tar -xf gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz
  ```
2. Always export the following environment variables before building:
  ```
  export TENSOROPT_TOOLCHAIN_DIR=${GCC_LINARO_PATH}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu
  export TENSOROPT_SYSROOT_DIR=${GCC_LINARO_PATH}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu
  export TENSOROPT_TARGET_TRIPLE=aarch64-linux-gnu
  ```
3. Add the following to the CMake options: `-DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/gcc-generic.cmake -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DComputeCpp_HOST_DIR=path/to/host/computecpp -DComputeCpp_DIR=path/to/target/computecpp`

### Building the library
Build the TensorOpt library with:
```
mkdir build
cd build
cmake <cmake_options> ..
make tensoropt
```

### Building and running the tests
Build and run the tests with:
```
make
ctest
```
If the tests have been cross-compiled and copied over, use `LD_PRELOAD` or `LD_LIBRARY_PATH` to specify the path to the dependencies: `libOpenCL.so`, `libComputeCpp.so`, `libtensoropt.so` and the backend library.
