# Copyright (C) Codeplay Software Limited.
add_library(tensoropt_common INTERFACE)
target_sources(tensoropt_common INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/backend_print.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/device.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/device.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/event.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/event.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/macro.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/memory.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/model.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp"
)
target_link_libraries(tensoropt_common INTERFACE tensoropt_interface)
