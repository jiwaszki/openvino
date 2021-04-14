// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>
#include <ie_input_info.hpp>

namespace py = pybind11;

class InferRequestWrapper : public InferenceEngine::InferRequest {
public:
    using InferenceEngine::InferRequest::InferRequest;

    // ~InferRequestWrapper() = default;

    InferenceEngine::ConstInputsDataMap _inputsInfo;
    InferenceEngine::ConstOutputsDataMap _outputsInfo;
};

void regclass_InferRequest(py::module m);
