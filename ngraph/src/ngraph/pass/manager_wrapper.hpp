    //*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <list>
#include <memory>
#include <typeinfo>
#include <vector>
#include <string>

#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/manager_state.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/pass/validate.hpp"

#include "ngraph/pass/constant_folding.hpp"

namespace ngraph
{
    namespace pass
    {
        class ManagerWrapper;
        class ManagerState;
    }
}

class NGRAPH_API ngraph::pass::ManagerWrapper : public ngraph::pass::Manager
{
public:
    ManagerWrapper();
    ~ManagerWrapper();

    void register_pass(std::string pass_name)
    {
        #define PUSH_PASS(STR_NAME, CLASS_NAME) ((pass_name == STR_NAME) ? push_pass<CLASS_NAME>() : 0);

        // switch (pass_name)
        // {
        // case "ReshapeElimination":
        //     rc = push_pass<T>(pass::ReshapeElimination);
        //     break;
        PUSH_PASS("ConstantFolding", pass::ConstantFolding)
        // if (pass_name == "")
        //     push_pass<>();
        //     break;
        // default:
        //     break;
        // }

        #undef PUSH_PASS

        if (m_per_pass_validation)
        {
            push_pass<Validate>();
        }
        return;
    }
};