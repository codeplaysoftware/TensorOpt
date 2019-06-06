# Copyright (C) Codeplay Software Limited.
find_package(IMGDNN REQUIRED)
add_library(tensoropt_public_backend INTERFACE)
target_link_libraries(tensoropt_public_backend INTERFACE
  IMGDNN::IMGDNN
)

add_library(tensoropt_private_backend INTERFACE)
target_sources(tensoropt_private_backend INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/convert.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/convert.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/compilation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/compilation.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/execution.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/execution.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/model.cpp"
)
target_link_libraries(tensoropt_private_backend INTERFACE
  tensoropt_common
)

add_library(imgdnn_network_binary_symbols "imgdnn_network_binary_symbols.cpp")
target_include_directories(imgdnn_network_binary_symbols PRIVATE ${IMGDNN_INCLUDE_DIR})
