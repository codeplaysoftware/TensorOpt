// Copyright (C) Codeplay Software Limited.
#ifndef TENSOROPT_TESTS_COMMON_TEST_UTILS_HPP
#define TENSOROPT_TESTS_COMMON_TEST_UTILS_HPP

#include <iostream>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <tensoropt/tensoropt.hpp>

#include "common/utils.hpp"

#define TENSOROPT_ASSERT_OK(VAR) ASSERT_EQ((VAR), ANEURALNETWORKS_NO_ERROR)

template <class Container>
typename Container::value_type totalSize(const Container& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<typename Container::value_type>());
}

#define ADD_TEST_HELPER(FIXTURE, TEST_NAME, FUNC) \
  TEST_F(FIXTURE, TEST_NAME) { FUNC(); }

#define ADD_TEST_HELPER_ARGS(FIXTURE, TEST_NAME, FUNC, ...) \
  TEST_F(FIXTURE, TEST_NAME) { FUNC(__VA_ARGS__); }

#endif  // TENSOROPT_TESTS_COMMON_TEST_UTILS_HPP
