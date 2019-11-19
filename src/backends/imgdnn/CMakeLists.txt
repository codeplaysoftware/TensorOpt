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
