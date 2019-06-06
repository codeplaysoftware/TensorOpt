# Copyright (C) Codeplay Software Limited.
add_library(tensoropt_common_test INTERFACE)
target_sources(tensoropt_common_test INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/common_fixture.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/test_utils.hpp"
)
target_include_directories(tensoropt_common_test INTERFACE
  # Re-use utility functions from the src directory too
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tests>
)
