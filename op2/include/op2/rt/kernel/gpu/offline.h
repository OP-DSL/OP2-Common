#pragma once

// The calls on what the application compiled, made from op2_kernels with the
// runtime that registered it: an offline kernel's attributes and launches,
// and the device addresses of its __constant__ symbols.  The generated code
// describes calls as its loops' (op2/rt/abi/gpu/descriptor.h).

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/kernel/prelude.h>

#include <cstdio>
#include <cstdlib>

// The runtime's names.
#if defined(__HIPCC__)
#define OP_KERNEL_RT(name) hip##name
#else
#include <cuda_runtime.h>
#define OP_KERNEL_RT(name) cuda##name
#endif

namespace op::rt::kernel::gpu {

namespace detail {

// Stop the program on a failed runtime call, naming it.
inline void check(OP_KERNEL_RT(Error_t) error, const char *call) {
    if (error == OP_KERNEL_RT(Success))
        return;
    std::fprintf(stderr, "error: %s failed with %s\n", call,
                 OP_KERNEL_RT(GetErrorString)(error));
    std::exit(1);
}

inline op::rt::abi::gpu::KernelResources resources(const void *function) {
    OP_KERNEL_RT(FuncAttributes) attributes{};
    check(OP_KERNEL_RT(FuncGetAttributes)(&attributes, function), "FuncGetAttributes");
    return {attributes.numRegs, static_cast<int>(attributes.localSizeBytes),
            static_cast<int>(attributes.sharedSizeBytes), attributes.maxThreadsPerBlock};
}

inline void allow_shared(const void *function, int bytes) {
    check(OP_KERNEL_RT(FuncSetAttribute)(
              function, OP_KERNEL_RT(FuncAttributeMaxDynamicSharedMemorySize), bytes),
          "FuncSetAttribute");
}

inline void launch(const void *function, int blocks, int block_size, void **args,
                   int shared_bytes) {
    check(OP_KERNEL_RT(LaunchKernel)(function, blocks, block_size, args, shared_bytes, 0),
          "LaunchKernel");
    check(OP_KERNEL_RT(PeekAtLastError)(), "PeekAtLastError");
}

inline void *symbol(const void *symbol) {
    void *address = nullptr;
    check(OP_KERNEL_RT(GetSymbolAddress)(&address, symbol), "GetSymbolAddress");
    return address;
}

} // namespace detail

inline constexpr op::rt::abi::gpu::OfflineCalls calls{&detail::resources, &detail::allow_shared,
                                                      &detail::launch, &detail::symbol};

} // namespace op::rt::kernel::gpu

#undef OP_KERNEL_RT
