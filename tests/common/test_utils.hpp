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
