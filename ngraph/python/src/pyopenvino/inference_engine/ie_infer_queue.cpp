// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <cpp/ie_executable_network.hpp>
#include <cpp/ie_infer_request.hpp>
#include <ie_common.h>
#include <ie_iinfer_request.hpp>

#include "pyopenvino/inference_engine/common.hpp"
#include "pyopenvino/inference_engine/ie_infer_request.hpp"
#include "pyopenvino/inference_engine/ie_infer_queue.hpp"

namespace py = pybind11;

class InferQueue
{
public:
    InferQueue(std::vector<InferRequestWrapper> requests,
               std::queue<size_t> idle_handles,
               std::vector<py::object> user_ids)
        : _requests(requests)
        , _idle_handles(idle_handles)
        , _user_ids(user_ids)
    {
        this->setDefaultCallbacks();
    }

    ~InferQueue() { _requests.clear(); }

    size_t getIdleRequestId()
    {
        // Wait for any of _idle_handles
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] { return !(_idle_handles.empty()); });

        size_t idle_request_id = _idle_handles.front();
        _idle_handles.pop();

        return idle_request_id;
    }

    std::vector<InferenceEngine::StatusCode> waitAll()
    {
        // Wait for all requests to return with callback thus updating
        // _idle_handles so it matches the size of requests
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> lock(_mutex);
        _cv.wait(lock, [this] { return _idle_handles.size() == _requests.size(); });

        std::vector<InferenceEngine::StatusCode> statuses;

        for (size_t handle = 0; handle < _requests.size(); handle++)
        {
            statuses.push_back(
                _requests[handle].Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY));
        }

        return statuses;
    }

    void setDefaultCallbacks()
    {
        for (size_t handle = 0; handle < _requests.size(); handle++)
        {
            _requests[handle].SetCompletionCallback([this, handle /* ... */]() {
                py::gil_scoped_acquire acquire;
                _idle_handles.push(handle);
                _cv.notify_one();
            });
        }
    }

    void setCustomCallbacks(py::function f_callback)
    {
        for (size_t handle = 0; handle < _requests.size(); handle++)
        {
            _requests[handle].SetCompletionCallback([this, f_callback, handle /* ... */]() {
                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                f_callback(_requests[handle], _user_ids[handle]);
                // Add idle handle to queue
                _idle_handles.push(handle);
                // Notify locks in getIdleRequestId() or waitAll() functions
                _cv.notify_one();
            });
        }
    }

    std::vector<InferRequestWrapper> _requests;
    std::vector<py::object> _user_ids; // user ID can be any Python object
    std::queue<size_t> _idle_handles;
    std::mutex _mutex;
    std::condition_variable _cv;
};

void regclass_InferQueue(py::module m)
{
    py::class_<InferQueue, std::shared_ptr<InferQueue>> cls(m, "InferQueue");

    cls.def(py::init([](InferenceEngine::ExecutableNetwork& net, size_t jobs) {
        std::vector<InferRequestWrapper> requests;
        std::queue<size_t> idle_handles;
        std::vector<py::object> user_ids(jobs);

        for (size_t handle = 0; handle < jobs; handle++)
        {
            auto request = static_cast<InferRequestWrapper>(net.CreateInferRequest());
            // Get Inputs and Outputs info from executable network
            request._inputsInfo = net.GetInputsInfo();
            request._outputsInfo = net.GetOutputsInfo();

            requests.push_back(request);
            idle_handles.push(handle);
        }

        return new InferQueue(requests, idle_handles, user_ids);
    }), py::arg("network"), py::arg("jobs") = 0);

    cls.def("_async_infer", [](InferQueue& self, const py::dict inputs, py::object userdata) {
        // getIdleRequestId function has an intention to block InferQueue
        // until there is at least one idle (free to use) InferRequest
        auto handle = self.getIdleRequestId();
        // Set new inputs label/id from user
        self._user_ids[handle] = userdata;
        // Update inputs of picked InferRequest
        if (!inputs.empty()) {
            Common::set_request_blobs(self._requests[handle], inputs);
        }
        // Now GIL can be released
        {
            py::gil_scoped_release release;
            // Start InferRequest in asynchronus mode
            self._requests[handle].StartAsync();
        }
    }, py::arg("inputs"), py::arg("userdata"));

    cls.def("wait_all", [](InferQueue& self) { return self.waitAll(); });

    cls.def("set_infer_callback",
            [](InferQueue& self, py::function f_callback) { self.setCustomCallbacks(f_callback); });

    cls.def("__len__", [](InferQueue& self) { return self._requests.size(); });

    cls.def(
        "__iter__",
        [](InferQueue& self) {
            return py::make_iterator(self._requests.begin(), self._requests.end());
        },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    cls.def("__getitem__", [](InferQueue& self, size_t i) { return self._requests[i]; });
}
