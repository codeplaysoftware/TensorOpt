/**
 * Copyright (C) Codeplay Software Limited.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INCLUDE_TENSOROPT_EVENT_HPP
#define INCLUDE_TENSOROPT_EVENT_HPP

#include <SYCL/sycl.hpp>

#include "tensoropt/result.hpp"

struct ANeuralNetworksEvent;

/**
 * Get a SYCL event.
 */
ResultCode ANeuralNetworksEvent_getSyclEvent(ANeuralNetworksEvent* event,
                                             cl::sycl::event* sycl_event);

/**
 * Wait on an event.
 * See ANeuralNetworksExecution_startCompute.
 */
ResultCode ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event);

/**
 * Free an event.
 * See ANeuralNetworksExecution_startCompute.
 */
void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event);

#endif  // INCLUDE_TENSOROPT_EVENT_HPP
