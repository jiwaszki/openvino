// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp/ie_infer_request.hpp>
#include <ie_blob.h>
#include <ie_parameter.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include "Python.h"
#include "ie_common.h"

namespace py = pybind11;

namespace Common
{
    template <typename T>
    const std::shared_ptr<InferenceEngine::TBlob<T>>
        create_blob_from_numpy(const py::handle& py_array, InferenceEngine::Precision precision)
    {
        py::array_t<T> arr = py::cast<py::array>(py_array);
        InferenceEngine::SizeVector dims;
        for (size_t i = 0; i < arr.ndim(); i++)
        {
            dims.push_back(arr.shape(i));
        }
        auto desc = InferenceEngine::TensorDesc(
            precision,
            dims,
            InferenceEngine::Layout::NCHW); // TODO: select Layout based on dims
        auto blob = InferenceEngine::make_shared_blob<T>(desc);
        blob->allocate();
        if (arr.size() != 0)
        {
            std::copy(
                arr.data(0), arr.data(0) + arr.size(), blob->rwmap().template as<T*>());
        }
        return blob;
    }

    InferenceEngine::Layout get_layout_from_string(const std::string& layout);

    const std::string& get_layout_from_enum(const InferenceEngine::Layout& layout);

    PyObject* parse_parameter(const InferenceEngine::Parameter& param);

    PyObject* parse_parameter(const InferenceEngine::Parameter& param);

    bool is_TBlob(const py::handle& blob);

    const std::shared_ptr<InferenceEngine::Blob> cast_to_blob(const py::handle& blob);

    const std::shared_ptr<InferenceEngine::Blob> blob_from_numpy(const py::handle& _arr);

    void set_request_blobs(InferenceEngine::InferRequest& request, const py::dict& dictonary);
}; // namespace Common
