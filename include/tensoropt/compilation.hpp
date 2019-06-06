// Copyright (C) Codeplay Software Limited.
#ifndef INCLUDE_TENSOROPT_COMPILATION_HPP
#define INCLUDE_TENSOROPT_COMPILATION_HPP

#include "tensoropt/model.hpp"

/**
 * Size of the token used for caching.
 * See ANeuralNetworksCompilation_setCaching.
 */
enum { BYTE_SIZE_OF_CACHE_TOKEN = 32 };

/**
 * Preference options when compiling a model.
 * See ANeuralNetworksCompilation_setPreference.
 */
enum PreferenceCode {
  ANEURALNETWORKS_PREFER_LOW_POWER = 0,
  ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
  ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2
};

/**
 * Compiles a ANeuralNetworksModel to an ANeuralNetworksExecution object.
 */
struct ANeuralNetworksCompilation;

/**
 * Create a ANeuralNetworksCompilation object from a model.
 * A Device will be created internally.
 */
ResultCode ANeuralNetworksCompilation_create(
    ANeuralNetworksModel* model, ANeuralNetworksCompilation** compilation);

/**
 * Create a ANeuralNetworksCompilation object from a model and a set of devices.
 */
ResultCode ANeuralNetworksCompilation_createForDevices(
    ANeuralNetworksModel* model, const ANeuralNetworksDevice* const* devices,
    uint32_t num_devices, ANeuralNetworksCompilation** compilation);

/**
 * Provide a cache directory to save and load compiled models.
 * The token must be a unique identifier for the model of size
 * BYTE_SIZE_OF_CACHE_TOKEN.
 * Caching if only used if this method is called and
 * ANeuralNetworksCompilation_serialize is used after. By default no caching is
 * performed.
 */
ResultCode ANeuralNetworksCompilation_setCaching(
    ANeuralNetworksCompilation* compilation, const char* cache_dir,
    const uint8_t* token);

ResultCode ANeuralNetworksCompilation_setPreference(
    ANeuralNetworksCompilation* compilation, int32_t preference);

/**
 * Mark the compilation as finished to be able to create an
 * ANeuralNetworksExecution object using ANeuralNetworksExecution_create.
 */
ResultCode ANeuralNetworksCompilation_finish(
    ANeuralNetworksCompilation* compilation);

/**
 * Serialize the compiled model, this replaces the call to finish.
 * The compilation should not be finished with ANeuralNetworksCompilation_finish
 * unless execution objects will be created using both
 * ANeuralNetworksExecution_create and
 * ANeuralNetworksExecution_createFromBinary.
 * See ANeuralNetworksExecution_createFromBinary to deserialize the
 * data.
 */
ResultCode ANeuralNetworksCompilation_serialize(
    ANeuralNetworksCompilation* compilation, void** data,
    std::size_t* data_size);

/**
 * Free a compilation.
 * A compilation cannot be free'd until all execution object created directly
 * from it are free'd too. This does not apply for execution object created from
 * serialized data.
 */
void ANeuralNetworksCompilation_free(ANeuralNetworksCompilation* compilation);

#endif  // INCLUDE_TENSOROPT_COMPILATION_HPP
