// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>

#include <string>

#include <ie_common.h>

#include "pyopenvino/inference_engine/common.hpp"
#include "pyopenvino/inference_engine/ie_executable_network.hpp"
#include "pyopenvino/inference_engine/ie_infer_request.hpp"
#include "pyopenvino/inference_engine/ie_preprocess_info.hpp"
#include "pyopenvino/inference_engine/containers.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferRequestWrapper, std::shared_ptr<InferRequestWrapper>> cls(
        m, "InferRequest");

    cls.def("set_batch", [](InferRequestWrapper& self, const int size) {
        self.SetBatch(size);
    }, py::arg("size"));

    cls.def("get_blob", [](InferRequestWrapper& self, const std::string& name) {
        return self.GetBlob(name);
    }, py::arg("name"));

    cls.def("set_blob", [](InferRequestWrapper& self,
                           const std::string& name,
                           py::handle& blob) {
        self.SetBlob(name, Common::cast_to_blob(blob));
    }, py::arg("name"), py::arg("blob"));

    cls.def("set_blob", [](InferRequestWrapper& self,
                           const std::string& name,
                           py::handle& blob,
                           const InferenceEngine::PreProcessInfo& info) {
        self.SetBlob(name, Common::cast_to_blob(blob));
    }, py::arg("name"), py::arg("blob"), py::arg("info"));

    cls.def("set_input", [](InferRequestWrapper& self, const py::dict& inputs) {
        Common::set_request_blobs(self, inputs);
    }, py::arg("inputs"));

    cls.def("set_output", [](InferRequestWrapper& self, const py::dict& results) {
        Common::set_request_blobs(self, results);
    }, py::arg("results"));

    cls.def("_infer", [](InferRequestWrapper& self, const py::dict& inputs) {
        // Update inputs if there are any
        if (!inputs.empty()) {
            Common::set_request_blobs(self, inputs);
        }
        // Call Infer function
        self.Infer();
        // Get output Blobs and return
        Containers::PyResults results;
        for (auto& out : self._outputsInfo)
        {
            results[out.first] = self.GetBlob(out.first);
        }
        return results;
    }, py::arg("inputs"));

    cls.def("_async_infer", [](InferRequestWrapper& self, const py::dict inputs) {
        if (!inputs.empty()) {
            Common::set_request_blobs(self._requests[handle], inputs);
        }
        py::gil_scoped_release release;
        self.StartAsync();
    });

    cls.def(
        "wait",
        [](InferRequestWrapper& self, int64_t millis_timeout) {
            py::gil_scoped_acquire acquire;
            return self.Wait(millis_timeout);
        },
        py::arg("millis_timeout") = InferenceEngine::IInferRequest::WaitMode::RESULT_READY);

    cls.def("set_completion_callback",
            [](InferRequestWrapper& self, py::function f_callback) {
                self.SetCompletionCallback([self, f_callback]() {
                    // py::gil_scoped_acquire acquire;
                    f_callback(self);
                    // py::gil_scoped_release release;
                });
            }, py::arg("f_callback"));

    cls.def("get_perf_counts", [](InferRequestWrapper& self) {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        perfMap = self.GetPerformanceCounts();
        py::dict perf_map;

        for (auto it : perfMap)
        {
            py::dict profile_info;
            switch (it.second.status)
            {
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                profile_info["status"] = "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                profile_info["status"] = "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                profile_info["status"] = "OPTIMIZED_OUT";
                break;
            default: profile_info["status"] = "UNKNOWN";
            }
            profile_info["exec_type"] = it.second.exec_type;
            profile_info["layer_type"] = it.second.layer_type;
            profile_info["cpu_time"] = it.second.cpu_uSec;
            profile_info["real_time"] = it.second.realTime_uSec;
            profile_info["execution_index"] = it.second.execution_index;
            perf_map[it.first.c_str()] = profile_info;
        }
        return perf_map;
    });

    cls.def("preprocess_info", [](InferRequestWrapper& self, const std::string& name) {
        return self.GetPreProcess(name);
    }, py::arg("name"));

    //    cls.def_property_readonly("preprocess_info", [](InferRequestWrapper& self) {
    //
    //    });

    cls.def_property_readonly("input_blobs", [](InferRequestWrapper& self) {
        Containers::PyResults input_blobs;
        for (auto& in : self._inputsInfo)
        {
            input_blobs[in.first] = self.GetBlob(in.first);
        }
        return input_blobs;
    });

    cls.def_property_readonly("output_blobs", [](InferRequestWrapper& self) {
        Containers::PyResults output_blobs;
        for (auto& out : self._outputsInfo)
        {
            output_blobs[out.first] = self.GetBlob(out.first);
        }
        return output_blobs;
    });

    // cls.def("__del__", [](InferRequestWrapper& self) {
    //     InferenceEngine::InferRequest::actual = nullptr;
    // });

    //    latency
}
