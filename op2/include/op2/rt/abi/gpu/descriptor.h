#pragma once

// A loop as the generated CUDA or HIP code describes it to the library, and
// the call it makes to run it: the kernels of each family it generated, the
// parameters they may be specialised on, the binding of their arguments,
// and the calls on what the application compiled (op2/rt/kernel/gpu/offline.h).

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/gpu/launch.h>
#include <op_lib_core.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace op::rt::abi::gpu {

// A parameter as the CUDA and HIP code registers it: what the JIT
// specialises on, and its device side.
struct Param : abi::Param {
    // The copy an offline kernel reads, a __constant__ symbol, if any, and
    // the hash of the value there, shared by every loop that reads it.
    const void *symbol = nullptr;
    std::uint64_t *shared_hash = nullptr;
    // Whether every kernel has the argument by value, needing no device copy.
    bool by_value = false;
};

template<typename T>
constexpr Param constant(const char *name, T *data, const void *symbol,
                         std::uint64_t *shared_hash) {
    return {abi::constant(name, data), symbol, shared_hash, false};
}

template<typename T>
constexpr Param constant(const char *name, T *data, std::size_t elems, const void *symbol,
                         std::uint64_t *shared_hash) {
    return {abi::constant(name, data, elems), symbol, shared_hash, false};
}

// Every kernel has a scalar READ global by value.
template<typename T>
constexpr Param scalar_global(const char *name, T *data, int arg) {
    return {abi::scalar_global(name, data, arg), nullptr, nullptr, true};
}

// Any other READ global argument: by value, or else read by an offline
// kernel from its device copy.
template<typename T>
constexpr Param global(const char *name, int arg, bool by_value) {
    return {abi::global<T>(name, arg), nullptr, nullptr, by_value};
}

constexpr Param dat_stride(const char *name, int *data, int arg, const void *symbol) {
    return {abi::dat_stride(name, data, arg), symbol, nullptr, false};
}

constexpr Param dat_dim(const char *name, int *data, int arg) {
    return {abi::dat_dim(name, data, arg), nullptr, nullptr, false};
}

// A kernel's resources as its compiler left them: registers and local
// memory per thread, static shared memory per block, and the most threads a
// block can have with them.
struct KernelResources {
    int registers = 0;
    int local_bytes = 0;
    int shared_bytes = 0;
    int max_threads = 0;
};

// The calls on a kernel or symbol compiled with the application, which
// op2_kernels makes with the runtime that registered it.
struct OfflineCalls {
    KernelResources (*resources)(const void *function);
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
    std::span<const char *const> data;
    std::span<const char *const> names;
};

// A strategy family the generated code built a kernel for: the kernel's
// name, its offline build if any, and the source of a specialised one.
struct Family : abi::Family {
    const char *kernel;
    const void *offline;
    const char *source;
};

struct Loop : abi::Loop {
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
};

// Run one call of loop.
void invoke(Loop& loop, op_set set, op_arg *args, int nargs);

} // namespace op::rt::abi::gpu
