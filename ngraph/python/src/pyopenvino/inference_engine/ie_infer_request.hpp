// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>

#include <pybind11/pybind11.h>

#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>
#include <ie_input_info.hpp>

namespace py = pybind11;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

class InferRequestWrapper : public InferenceEngine::InferRequest {
public:
    using InferenceEngine::InferRequest::InferRequest;

    // ~InferRequestWrapper() = default;

    // bool user_callback_defined;
    // py::function user_callback;
    InferenceEngine::ConstInputsDataMap _inputsInfo;
    InferenceEngine::ConstOutputsDataMap _outputsInfo;
    Time::time_point _startTime;
    Time::time_point _endTime;
};

void regclass_InferRequest(py::module m);
