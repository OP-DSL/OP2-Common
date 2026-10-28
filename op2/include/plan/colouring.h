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
// starts.  The colours below core_colours hold the elements before the halo
// wait only, so they run while the halo arrives, and those below
// owned_colours the elements up to the end of the section the globals
// follow, so a global reduction can be processed after them.
struct SetColouring {
    int colours = 0;
    int core_colours = 0;
    int owned_colours = 0;
    std::vector<int> offsets;
    std::vector<int> order;
};

// Colour each section's elements apart, greedily in element order, a
// section's colours after the earlier sections'.
SetColouring colour_set(std::span<const op_arg> args, const op::loop::Sections& sections);

} // namespace op::plan
