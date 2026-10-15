#pragma once

// The layout of a chunk partition as the host planner writes it and a staged
// kernel reads it.  Compiled on the host and for the device, by NVRTC too,
// so it includes nothing.

namespace op::chunk {

/* Staging, laid out as op_plan lays out a block: a chunk stages each staged
   dat's distinct targets in ascending order, a scratch slot each (op_plan's
   ind_map), and a stage word per staged argument and source element names
   its target's slot (op_plan's loc_map).  A region holds one dat's slots,
   each slot its dim components contiguous, aligned to the dat's scalar
   size. */
using StageWord = unsigned short;

// Slots a region may hold; the word past the last slot marks a reference
// left in global memory.
constexpr int max_slots = 0xffff;
constexpr StageWord unstaged_word = 0xffff;

} // namespace op::chunk
