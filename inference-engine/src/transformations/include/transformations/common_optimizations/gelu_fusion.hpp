// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API GeluFusion;
class TRANSFORMATIONS_API GeluFusionWithErfOne;
class TRANSFORMATIONS_API GeluFusionWithErfTwo;
class TRANSFORMATIONS_API GeluFusionWithErfThree;

} // namespace pass
} // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph 0.5 * (x * (1 + erf(x /
 * sqrt(2))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfOne : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  GeluFusionWithErfOne();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph 0.5 * (x * (1 + erf(x /
 * sqrt(2))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfTwo : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  GeluFusionWithErfTwo();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces a sub-graph 0.5 * (x * (1 + erf(x /
 * sqrt(2))) with a Gelu op.
 */
class ngraph::pass::GeluFusionWithErfThree : public ngraph::pass::MatcherPass {
public:
  NGRAPH_RTTI_DECLARATION;
  GeluFusionWithErfThree();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief GeluFusion transformation replaces various sub-graphs with a Gelu op.
 */
class ngraph::pass::GeluFusion : public ngraph::pass::GraphRewrite {
public:
  NGRAPH_RTTI_DECLARATION;
  GeluFusion() {
    add_matcher<ngraph::pass::GeluFusionWithErfOne>();
    add_matcher<ngraph::pass::GeluFusionWithErfTwo>();
    add_matcher<ngraph::pass::GeluFusionWithErfThree>();
  }
};
