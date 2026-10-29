#pragma once

// A loop as the generated CUDA or HIP code describes it to the library, and
// the call it makes to run it: the kernels of each family it generated, the
// parameters they may be specialised on, the binding of their arguments,
// and the calls on what the application compiled (op2/rt/kernel/gpu/offline.h).

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/gpu/launch.h>
#include <op_lib_core.h>

#include <span>
#include <vector>

namespace op::rt::abi::gpu {

// The calls on a kernel or symbol compiled with the application, which
// op2_kernels makes with the runtime that registered it.
struct OfflineCalls {
    KernelFigures (*figures)(const void *function);
    // Let a launch have up to bytes of dynamic shared memory.
    void (*allow_shared)(const void *function, int bytes);
    void (*launch)(const void *function, int blocks, int block_size, void **args,
                   int shared_bytes);
    // A __constant__ symbol's device address.
    void *(*symbol)(const void *symbol);
};

// Append a launch's kernel arguments after its record's fields, the same for
// every family's kernels: the loop's dats, maps, globals and indices; and
// mark its active optional arguments in opt_flags.
using Binder = void (*)(op_arg *args, unsigned& opt_flags, std::vector<void *>& arguments);

// The headers a specialised kernel's source includes, in order, embedded in
// the translation unit (op2/rt/kernel/gpu/jit_embed.h).
struct JitHeaders {
    std::vector<const char *> data;
    std::vector<const char *> names;
};

// A strategy family the generated code built a kernel for: the kernel's
// name, its offline build if any, and the source of a specialised one; the
// chunked families' groups are the arguments they stage.
struct Family {
    const char *name;
    const char *kernel;
    const void *offline;
    const char *source;
    ArgGroups groups;
};

struct Loop {
    const char *name;
    // The loop to the profiler.
    const char *target;
    const char *variant;
    bool direct;
    // A global OP_INC is incremented atomically, in place, rather than
    // reduced from a copy per thread.
    bool gbl_inc_atomic;
    // Whether the kernels keep a copy per thread of some global, so they read
    // the global stride.
    bool per_thread_globals;
    std::span<const Param> params;
    std::span<const Family> families;
    Binder bind;
    const OfflineCalls *calls;
    JitHeaders headers;
    // The library's loop, made on the first call.
    void *runtime = nullptr;
};

// Run one call of loop.
void invoke(Loop& loop, op_set set, op_arg *args, int nargs);

} // namespace op::rt::abi::gpu
