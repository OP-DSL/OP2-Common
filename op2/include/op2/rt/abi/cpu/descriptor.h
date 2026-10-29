#pragma once

// A loop as the generated OpenMP code describes it to the library, and the
// call it makes to run it: the region each family's phases run in, and the
// host's thread count.

#include <op2/rt/abi/cpu/launch.h>
#include <op2/rt/abi/descriptor.h>
#include <op_lib_core.h>

#include <span>

namespace op::rt::abi::cpu {

// Run a phase of a loop in the generated code's OpenMP region.
using RangeRun = void (*)(const Range& range, op_arg *args);
using ChunkRun = void (*)(const Chunk& chunks, op_arg *args);

// A strategy family the generated code built a region for: the plain,
// atomics and colouring families' over a range, the chunked families' over
// chunks, the other null.
struct Family : abi::Family {
    RangeRun range;
    ChunkRun chunks;
};

struct Loop : abi::Loop {
    std::span<const Family> families;
    // The generated code's omp_get_max_threads.
    int (*max_threads)();
};

// Run one call of loop.
void invoke(Loop& loop, op_set set, op_arg *args, int nargs);

} // namespace op::rt::abi::cpu
