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
from ngraph.impl import AxisSet, Function, Shape, Type
from ngraph.impl.op import Constant, Parameter
from tests.runtime import get_runtime


def test_constant_folding_ceiling():
    auto constant = op::Constant::create(
        element::f32, Shape{2, 3}, vector<float>{0.0f, 0.1f, -0.1f, -2.5f, 2.5f, 3.0f});
    auto ceil = make_shared<op::Ceiling>(constant);
    auto f = make_shared<Function>(ceil, ParameterVector{});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Ceiling>(f), 0);
    ASSERT_EQ(count_ops_of_type<op::Constant>(f), 1);

    auto new_const = as_type_ptr<op::Constant>(f->get_results().at(0)->get_argument(0));
    ASSERT_TRUE(new_const);
    auto values_out = new_const->get_vector<float>();

    vector<float> values_expected{0.0f, 1.0f, 0.0f, -2.0f, 3.0f, 3.0f};

    ASSERT_TRUE(test::all_close_f(values_out, values_expected, MIN_FLOAT_TOLERANCE_BITS));
