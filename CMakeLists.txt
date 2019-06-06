# Copyright (C) Codeplay Software Limited.
cmake_minimum_required(VERSION 3.4.3)

project(TensorOpt)

set(TENSOROPT_BACKEND "" CACHE STRING "Backend to compile")
# The list of backends below should match the subdirectories of src/backends
set_property(CACHE TENSOROPT_BACKEND PROPERTY STRINGS "IMGDNN")

option(TENSOROPT_VERBOSE_LOG "Enable verbose logging for debugging purposes" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

set_property(TARGET OpenCL::OpenCL
             APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS
             CL_TARGET_OPENCL_VERSION=120)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
foreach(flag -Wall -Wextra -Wpedantic -Wconversion)
  check_cxx_compiler_flag(${flag} is_flag_supported)
  if(is_flag_supported)
    add_compile_options(${flag})
  endif()
endforeach()

add_library(tensoropt_interface INTERFACE)
target_sources(tensoropt_interface INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/compilation.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/device.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/event.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/execution.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/memory.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/model.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/operand.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/operation.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/result.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tensoropt/tensoropt.hpp"
)
target_include_directories(tensoropt_interface INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
set_property(TARGET tensoropt_interface PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tensoropt_interface INTERFACE ComputeCpp::ComputeCpp)

if(TENSOROPT_VERBOSE_LOG)
  set_property(TARGET tensoropt_interface PROPERTY INTERFACE_COMPILE_DEFINITIONS VERBOSE_LOG=1)
endif()

add_subdirectory(src/common)
string(TOLOWER ${TENSOROPT_BACKEND} backend_subdir)
set(backend_subdir "${CMAKE_CURRENT_SOURCE_DIR}/src/backends/${backend_subdir}")
if(EXISTS ${backend_subdir})
  message(STATUS "Using ${TENSOROPT_BACKEND} backend")
  add_subdirectory(${backend_subdir})
else()
  if(TENSOROPT_BACKEND STREQUAL "")
    message(FATAL_ERROR "No backend provided, use -DTENSOROPT_BACKEND to set one.")
  else()
    message(FATAL_ERROR "Unknown backend '${TENSOROPT_BACKEND}'")
  endif()
endif()

add_library(tensoropt "")
target_link_libraries(tensoropt
  PRIVATE tensoropt_private_backend
  PUBLIC tensoropt_public_backend
  PUBLIC ComputeCpp::ComputeCpp
)
target_include_directories(tensoropt PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Install library
install(TARGETS tensoropt
        RUNTIME DESTINATION lib COMPONENT libraries
        LIBRARY DESTINATION lib COMPONENT libraries
        ARCHIVE DESTINATION lib COMPONENT libraries)

# Install headers
install(DIRECTORY include
        DESTINATION .
        COMPONENT headers
        FILES_MATCHING
          PATTERN "*.hpp"
          PATTERN "*.h")
