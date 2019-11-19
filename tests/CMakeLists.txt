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
include(GTest)
add_subdirectory(common)

include(CMakeParseArguments)
function(add_tensoropt_executable)
  cmake_parse_arguments("ARG" "" "TARGET" "SOURCES" ${ARGN})
  add_executable(${ARG_TARGET} ${ARG_SOURCES})
  target_link_libraries(${ARG_TARGET} PRIVATE tensoropt)
endfunction()

function(add_tensoropt_gtest)
  cmake_parse_arguments("ARG" "" "TARGET" "SOURCES" ${ARGN})
  add_tensoropt_executable(${ARGN})
  target_link_libraries(${ARG_TARGET} PRIVATE
    tensoropt_common_test
    gtest_main
  )
  add_test(NAME ${ARG_TARGET} COMMAND ${ARG_TARGET})
endfunction()

add_subdirectory(basic_sample)
add_subdirectory(test_operations)
add_subdirectory(test_serialize)
