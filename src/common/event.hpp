// Copyright (C) Codeplay Software Limited.
#ifndef SRC_COMMON_EVENT_HPP
#define SRC_COMMON_EVENT_HPP

#include <SYCL/sycl.hpp>

#include "tensoropt/event.hpp"
#include "tensoropt/execution.hpp"

struct ANeuralNetworksEvent {
  ANeuralNetworksEvent(const cl::sycl::event& sycl_event_,
                       ANeuralNetworksExecution* execution_)
      : sycl_event(sycl_event_), execution(execution_) {}

  ANeuralNetworksEvent(const ANeuralNetworksEvent&) = default;
  ANeuralNetworksEvent(ANeuralNetworksEvent&&) = default;

  ANeuralNetworksEvent& operator=(const ANeuralNetworksEvent&) = default;
  ANeuralNetworksEvent& operator=(ANeuralNetworksEvent&&) = default;

  cl::sycl::event sycl_event;
  ANeuralNetworksExecution* execution;  // weak_ptr
};

#endif  // SRC_COMMON_EVENT_HPP
