// Copyright (C) Codeplay Software Limited.
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
