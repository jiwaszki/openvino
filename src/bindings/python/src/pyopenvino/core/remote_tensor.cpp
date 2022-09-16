// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_tensor.hpp"

#include <pybind11/stl.h>

#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_RemoteTensor(py::module m) {
    py::class_<ov::RemoteTensor, std::shared_ptr<ov::RemoteTensor>, ov::Tensor> cls(m, "RemoteTensor");
    cls.doc() = "openvino.runtime.RemoteTensor holding memory on specific device, remote version of Tensor.";

    // Note:
    // No constructors provided, RemoteTensor must be created from given RemoteContext within Core instance

    // Class specific methods
    cls.def(
        "get_params",
        [](ov::RemoteTensor& self) {
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
            &ov::RemoteTensor::get_device_name,
            R"(
            Gets name of a device on which the underlying object is allocated.

            :return: A device name string in fully specified format
                     `<device_name>[.<device_id>[.<tile_id>]]`.
            :rtype: str
        )");

    // Inherited class memebers
    cls.def("get_element_type",
            &ov::RemoteTensor::get_element_type,
            R"(
            Gets RemoteTensor's element type.

            :rtype: openvino.runtime.Type
            )");

    cls.def_property_readonly("element_type",
                              &ov::RemoteTensor::get_element_type,
                              R"(
                                RemoteTensor's element type.

                                :rtype: openvino.runtime.Type
                              )");

    cls.def("get_size",
            &ov::RemoteTensor::get_size,
            R"(
            Gets RemoteTensor's size as total number of elements.

            :rtype: int
            )");

    cls.def_property_readonly("size",
                              &ov::RemoteTensor::get_size,
                              R"(
                                RemoteTensor's size as total number of elements.

                                :rtype: int
                              )");

    cls.def("get_byte_size",
            &ov::RemoteTensor::get_byte_size,
            R"(
            Gets RemoteTensor's size in bytes.

            :rtype: int
            )");

    cls.def_property_readonly("byte_size",
                              &ov::RemoteTensor::get_byte_size,
                              R"(
                                RemoteTensor's size in bytes.

                                :rtype: int
                              )");

    cls.def("get_strides",
            &ov::RemoteTensor::get_strides,
            R"(
            Gets RemoteTensor's strides in bytes.

            :rtype: openvino.runtime.Strides
            )");

    cls.def_property_readonly("strides",
                              &ov::RemoteTensor::get_strides,
                              R"(
                                RemoteTensor's strides in bytes.

                                :rtype: openvino.runtime.Strides
                              )");

    cls.def("get_shape",
            &ov::RemoteTensor::get_shape,
            R"(
            Gets RemoteTensor's shape.

            :rtype: openvino.runtime.Shape
            )");

    cls.def("set_shape",
            &ov::RemoteTensor::set_shape,
            R"(
            Sets RemoteTensor's shape.
            )");

    cls.def(
        "set_shape",
        [](ov::RemoteTensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            Sets RemoteTensor's shape.
        )");

    cls.def_property("shape",
                     &ov::RemoteTensor::get_shape,
                     &ov::RemoteTensor::set_shape,
                     R"(
                        RemoteTensor's shape get/set.
                     )");

    cls.def_property(
        "shape",
        &ov::RemoteTensor::get_shape,
        [](ov::RemoteTensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            RemoteTensor's shape get/set.
        )");

    cls.def("__repr__", [](const ov::RemoteTensor& self) {
        std::stringstream ss;

        ss << "shape" << self.get_shape() << " type: " << self.get_element_type();

        return "<Tensor: " + ss.str() + ">";
    });

    // Removed function in this class, should throw error
    cls.def_property_readonly(
        "data",
        [](ov::RemoteTensor& self) {
            throw ov::Exception("Access to host memory is not available for RemoteTensor!");
        },
        R"(
            Access to host memory is not available for RemoteTensor.
            To access a device-specific memory, cast to a specific RemoteTensor
            derived object and work with its properties or parse device memory
            properties via `RemoteTensor.get_params()`.
        )");
}
