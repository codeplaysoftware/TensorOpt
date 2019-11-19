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
#ifndef SRC_COMMON_BACKEND_PRINT_HPP
#define SRC_COMMON_BACKEND_PRINT_HPP

#include <iostream>
#include <string>

#include "common/macro.hpp"
#include "common/utils.hpp"

// Need to disambiguate nullptr
inline std::ostream& operator<<(std::ostream& os, std::nullptr_t) {
  return os << static_cast<void*>(nullptr);
}

// Utility function to print some specific pointers the same way.
// This is not an operator<< overload to avoid ambiguity.
template <class T>
inline std::ostream& printPointer(std::ostream& os, T* ptr) {
  if (ptr) {
    return os << "&" << *ptr;
  }
  return os << ptr;
}

template <class T>
inline void printHostData(const T& data, std::size_t count) {
  if (count == 0) {
    VLOG("nullptr");
  } else {
    VLOG("&{" << arrayToString(data, count) << "}");
  }
}

// A backend can specialize the print of a specific argument
template <class Arg>
inline void backendPrintArg(Arg&& arg) {
  TENSOROPT_UNUSED_VARIABLE(arg);
  VLOG(arg);
}

template <class T, unsigned N>
inline void backendPrintArg(T (&arg)[N]) {
  TENSOROPT_UNUSED_VARIABLE(arg);
  printHostData(arg, N);
}

inline void backendPrintArgs() {}

template <class Arg>
inline void backendPrintArgs(Arg&& arg) {
  backendPrintArg(arg);
}

template <class Arg, class... Args>
void backendPrintArgs(Arg&& arg, Args&&... args) {
  backendPrintArg(arg);
  VLOG(", ");
  backendPrintArgs(args...);
}

// A backend can specialize the print of a specific function
template <class FuncPtr, class... Args>
struct BackendPrintFunc {
  inline void operator()(const std::string& func_name, Args&&... args) const {
    TENSOROPT_UNUSED_VARIABLE(func_name);
    VLOG(func_name << "(");
    backendPrintArgs(args...);
    VLOG(")");
  }
};

// Helper function to deduce BackendPrintFunc template arguments
template <class FuncPtr, class... Args>
inline void backendPrintFunc(const std::string& func_name, FuncPtr,
                             Args&&... args) {
  BackendPrintFunc<FuncPtr, Args...>()(func_name, std::forward<Args>(args)...);
}

#ifdef VERBOSE_LOG
// Print FUNC and its arguments, execute FUNC(...)
#define BACKEND_CALL(FUNC, ...)                 \
  {                                             \
    backendPrintFunc(#FUNC, FUNC, __VA_ARGS__); \
    VLOG(std::endl);                            \
    FUNC(__VA_ARGS__);                          \
  }

// Print FUNC and its arguments, execute FUNC(...) and print its return value
#define BACKEND_CALL_RET(RET, FUNC, ...)        \
  {                                             \
    backendPrintFunc(#FUNC, FUNC, __VA_ARGS__); \
    (RET) = FUNC(__VA_ARGS__);                  \
    VLOG(" -> ");                               \
    backendPrintArg(RET);                       \
    VLOG(std::endl);                            \
  }
#else  // VERBOSE_LOG
// Execute FUNC(...)
#define BACKEND_CALL(FUNC, ...) \
  { FUNC(__VA_ARGS__); }

// Execute FUNC(...) and return its value
#define BACKEND_CALL_RET(RET, FUNC, ...) \
  { (RET) = FUNC(__VA_ARGS__); }
#endif

#endif  // SRC_COMMON_BACKEND_PRINT_HPP
