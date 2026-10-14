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

/* Run-wide options of the chunked strategies, from OP_CHUNK_ATOMICS_* and
   OP_CHUNK_COLOURING_* variables.  The planner reads them from its options
   and a JIT kernel as one constant, so a kernel holds only the code its
   options need. */
enum Variant : int {
    // chunk_atomics: a chunk's only reference to a target increments global
    // memory directly, as staging it would save no global atomic.
    variant_unstaged_singles = 1 << 0,
    // chunk_colouring: one launch per section, its chunks taken in an order
    // that keeps them after every earlier chunk sharing a target.
    variant_wavefront = 1 << 1,
};

} // namespace op::chunk
