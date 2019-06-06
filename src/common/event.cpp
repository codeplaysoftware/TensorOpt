// Copyright (C) Codeplay Software Limited.
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
