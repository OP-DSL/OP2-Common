#pragma once

// A loop as the generated OpenMP code describes it to the library, and the
// call it makes to run it: the region each family's phases run in, and the
// host's thread count.

#include <op2/rt/abi/cpu/launch.h>
#include <op2/rt/abi/descriptor.h>
#include <op_lib_core.h>

#include <span>

namespace op::rt::abi::cpu {

// A strategy family the generated code built a body for, run through its
// OpenMP region; chunk_atomics' groups are the arguments it stages.
struct Family {
    const char *name;
    Run run;
    ArgGroups groups;
};

struct Loop {
    const char *name;
    // The loop to the profiler.
    const char *target;
    const char *variant;
    bool direct;
    std::span<const Family> families;
    // The generated code's omp_get_max_threads.
    int (*max_threads)();
    // The library's loop, made on the first call.
    void *runtime = nullptr;
};

// Run one call of loop.
void invoke(Loop& loop, op_set set, op_arg *args, int nargs);

} // namespace op::rt::abi::cpu
