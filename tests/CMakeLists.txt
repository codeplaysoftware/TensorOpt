# Copyright (C) Codeplay Software Limited.
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
