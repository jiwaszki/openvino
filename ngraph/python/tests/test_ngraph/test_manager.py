# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
# flake8: noqa
import numpy as np

import ngraph as ng
from ngraph.impl import AxisSet, Function, Shape, Type, Manager
from ngraph.impl.op import Constant, Parameter
from tests.runtime import get_runtime


def test_constant_folding_ceiling():
    input_array = [0.0, 0.1, -0.1, -2.5, 2.5, 3.0]
    node_constant = ng.constant(input_array, dtype=np.float32) # TODO: Shape{2, 3}
    node_ceil = ng.ceiling(constant)
    func = Function(node_ceil, ParameterVector{}) # TODO: ParameterVector
    # result = run_op_node(input_array, ng_api_fn)

    pass_manager = Manager()
    pass_manager.register_pass<ConstantFolding>()
    pass_manager.run_passes(f)

    assert count_ops_of_type<op::Ceiling>(f) == 0
    assert count_ops_of_type<op::Constant>(f) == 1

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    assert new_const
    auto values_out = new_const.get_vector();

    values_expected = [0.0, 1.0, 0.0, -2.0, 3.0, 3.0]

    assert np.all_close(values_out, values_expected)
