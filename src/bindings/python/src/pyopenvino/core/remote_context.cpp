// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_context.hpp"

#include <pybind11/stl.h>

#include "openvino/runtime/remote_context.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_RemoteContext(py::module m) {
    py::class_<ov::RemoteContext, std::shared_ptr<ov::RemoteContext>> cls(m, "RemoteContext");
    cls.doc() = "openvino.runtime.RemoteContext is remote (non-CPU) accelerator device-specific inference context.";

    // Note:
    // RemoteContext must be created from given user Context or Core instance,
    // empty constructor is provided to match API
    cls.def(py::init<>());

    cls.def(
        "get_params",
        [](ov::RemoteContext& self) {
            ov::AnyMap tmp_params = self.get_params();

            py::dict converted_params{};
            for (auto const& param : tmp_params) {
                converted_params[py::cast(param.first)] = Common::utils::from_ov_any(param.second);
            }

            return converted_params;
        },
        R"(
            Gets a dictionary of device-specific parameters required for low-level
            operations with underlying object.
            Parameters include device/context/surface/buffer handles, access flags, etc.
            Content of the returned dictionary depends on remote execution context that is
            currently set on the device (working scenario).

            :return: A dictionary of name/parameter elements.
            :rtype: dict
        )");

    cls.def("get_device_name",
            &ov::RemoteContext::get_device_name,
            R"(
            Gets name of a device on which the underlying object is allocated.

            :return: A device name string in fully specified format
                     `<device_name>[.<device_id>[.<tile_id>]]`.
            :rtype: str
        )");

    // TODO: overloads for "any-cl" based object, use void* or py::object and cast?
    cls.def(
        "create_tensor",
        [](ov::RemoteContext& self,
           const ov::element::Type& type,
           const ov::Shape& shape,
           const ov::AnyMap& params = {}) {
            throw ov::Exception("Not implemented!");
        },
        R"(
            Allocates memory tensor in device memory or wraps user-supplied
            memory handle using the specified tensor description and low-level
            device-specific parameters.

            :return: Plugin object that implements RemoteTensor.
            :rtype: openvino.runtime.RemoteTensor
        )");

    // TODO: add overloads for types (dtypes) and shapes
    cls.def(
        "create_host_tensor",
        [](ov::RemoteContext& self, const ov::element::Type& type, const ov::Shape& shape) {
            return self.create_host_tensor(type, shape);
        },
        R"(
            This method is used to create a host tensor object friendly for
            the device in current context. For example, GPU context may allocate
            USM host memory (if corresponding extension is available),
            which could be more efficient than regular host memory.

            :param type: Tensor element type.
            :type type: openvino.runtime.Type
            :param shape: Tensor shape.
            :type shape: openvino.runtime.Shape
            :return: A tensor instance with device friendly memory.
            :rtype: openvino.runtime.Tensor
        )");
}
