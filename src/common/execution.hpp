// Copyright (C) Codeplay Software Limited.
#ifndef SRC_COMMON_EXECUTION_HPP
#define SRC_COMMON_EXECUTION_HPP

#include "tensoropt/execution.hpp"

/**
 * Some backends may need to be notified of a wait, for instance
 * to perform the copy of any host output.
 */
ResultCode ANeuralNetworksExecution_notifyWait(
    ANeuralNetworksExecution* execution);

#endif  // SRC_COMMON_EXECUTION_HPP
