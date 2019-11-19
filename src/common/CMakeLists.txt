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
