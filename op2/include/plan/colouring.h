#pragma once

// Colour a set's elements so that no two of a colour update the same target
// of any indirect argument, as op_plan did under OP_COLOR2.  Host code with
// no device knowledge; a strategy runs one phase per colour.

#include <op_lib_core.h>

#include <span>
#include <vector>

namespace op::plan {

// order lists the elements colour by colour, offsets where each colour's run
// starts, and the colours below owned_colours hold owned elements only, so a
// global reduction can be processed after them.
struct SetColouring {
    int colours = 0;
    int owned_colours = 0;
    std::vector<int> offsets;
    std::vector<int> order;
};

// Colour the elements the loop runs (the exec halo too when an indirect
// argument updates) greedily in element order; with separate_owned the owned
// elements take the first colours and the halo's its own after them.
SetColouring colour_set(op_set set, std::span<const op_arg> args,
                        bool separate_owned);

} // namespace op::plan
