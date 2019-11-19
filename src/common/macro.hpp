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
#ifndef SRC_COMMON_MACRO_HPP
#define SRC_COMMON_MACRO_HPP

#include "tensoropt/result.hpp"

#ifdef VERBOSE_LOG
#include <iostream>
#define VLOG(STR) std::cerr << STR
#else
#define VLOG(STR)
#endif

#define VLOG_ENDL(STR) VLOG(STR << std::endl)

#define VLOG_AT(STR) VLOG_ENDL(STR << "\n  at " << __FILE__ << ":" << __LINE__)

#define TENSOROPT_UNUSED_VARIABLE(VAR) (void) (VAR)

#define TENSOROPT_RETURN_IF_COND(COND, MSG, RET) \
  if (COND) {                                    \
    VLOG_AT(MSG);                                \
    return RET;                                  \
  }

#define TENSOROPT_RETURN_IF_NULL(VAR)                                         \
  TENSOROPT_RETURN_IF_COND(                                                   \
      (VAR) == nullptr, "Error: unexpected null argument \"" << #VAR << "\"", \
      ANEURALNETWORKS_UNEXPECTED_NULL)

#define TENSOROPT_RETURN_IF_ERROR(VAR)                              \
  {                                                                 \
    ResultCode res_code = (VAR);                                    \
    TENSOROPT_RETURN_IF_COND(res_code != ANEURALNETWORKS_NO_ERROR,  \
                             "Error code: " << res_code, res_code); \
  }

#define TENSOROPT_RETURN_IF_FINISHED(VAR)                                  \
  TENSOROPT_RETURN_IF_COND((VAR)->finished,                                \
                           "Error: " << #VAR << " is in a finished state", \
                           ANEURALNETWORKS_BAD_STATE)

#define TENSOROPT_RETURN_IF_UNFINISHED(VAR)                                    \
  TENSOROPT_RETURN_IF_COND(!(VAR)->finished,                                   \
                           "Error: " << #VAR << " is not in a finished state", \
                           ANEURALNETWORKS_BAD_STATE)

#define TENSOROPT_RETURN_IF_UNEXPECTED_MIN_SIZE(OP, NAME, MIN_SIZE)         \
  TENSOROPT_RETURN_IF_COND((OP.NAME).size() < MIN_SIZE,                     \
                           "Error: Expected at least " << MIN_SIZE << #NAME \
                                                       << " but got "       \
                                                       << (OP.NAME).size(), \
                           ANEURALNETWORKS_OP_FAILED)

#define TENSOROPT_RETURN_IF_UNEXPECTED_MAX_SIZE(OP, NAME, MAX_SIZE)        \
  TENSOROPT_RETURN_IF_COND((OP.NAME).size() > MAX_SIZE,                    \
                           "Error: Expected at most " << MAX_SIZE << #NAME \
                                                      << " but got "       \
                                                      << (OP.NAME).size(), \
                           ANEURALNETWORKS_OP_FAILED)

#define TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(OP, NAME, MIN_SIZE, \
                                                   MAX_SIZE)           \
  TENSOROPT_RETURN_IF_UNEXPECTED_MIN_SIZE(OP, NAME, MIN_SIZE)          \
  TENSOROPT_RETURN_IF_UNEXPECTED_MAX_SIZE(OP, NAME, MAX_SIZE)

#define TENSOROPT_RETURN_IF_UNEXPECTED_SIZE(OP, NAME, SIZE) \
  TENSOROPT_RETURN_IF_UNEXPECTED_MINMAX_SIZE(OP, NAME, SIZE, SIZE)

#define TENSOROPT_TO_UINT32_INDEX(INT32_IDX, UINT32_IDX)                       \
  TENSOROPT_RETURN_IF_COND(                                                    \
      INT32_IDX < 0, "Error: expected positive index but got " << (INT32_IDX), \
      ANEURALNETWORKS_BAD_DATA)                                                \
  uint32_t UINT32_IDX = static_cast<uint32_t>(INT32_IDX)

#endif  // SRC_COMMON_MACRO_HPP
