// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

using Type = ::testing::Types<op::v1::ReduceMin>;
INSTANTIATE_TYPED_TEST_CASE_P(type_prop_reduce_min, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_CASE_P(type_prop_reduce_min_et, ReduceArithmeticTest, Type);
