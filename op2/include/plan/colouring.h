#pragma once

// Colour a set's elements so that no two of a colour update the same target
// of any indirect argument, as op_plan did under OP_COLOR2.  Host code with
// no device knowledge; a strategy runs one phase per colour.

#include <loop/strategy.h>
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

// Colour the elements the sections run greedily in element order: those up
// to the end of the section the globals follow first, as the owned colours,
// then the rest.
SetColouring colour_set(std::span<const op_arg> args, const op::loop::Sections& sections);

} // namespace op::plan
