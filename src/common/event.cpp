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
#include "common/event.hpp"
#include "common/execution.hpp"
#include "common/macro.hpp"

ResultCode ANeuralNetworksEvent_getSyclEvent(ANeuralNetworksEvent* event,
                                             cl::sycl::event* sycl_event) {
  TENSOROPT_RETURN_IF_NULL(event);
  *sycl_event = event->sycl_event;
  return ANEURALNETWORKS_NO_ERROR;
}

ResultCode ANeuralNetworksEvent_wait(ANeuralNetworksEvent* event) {
  TENSOROPT_RETURN_IF_NULL(event);
  try {
    event->sycl_event.wait_and_throw();
  } catch (const cl::sycl::exception& e) {
    TENSOROPT_UNUSED_VARIABLE(e);
    VLOG_ENDL(e.what());
    return ANEURALNETWORKS_BAD_STATE;
  }
  TENSOROPT_RETURN_IF_ERROR(
      ANeuralNetworksExecution_notifyWait(event->execution));
  return ANEURALNETWORKS_NO_ERROR;
}

void ANeuralNetworksEvent_free(ANeuralNetworksEvent* event) {
  if (event) {
    delete event;
  }
}
