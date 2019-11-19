#  Copyright (C) Codeplay Software Limited.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
