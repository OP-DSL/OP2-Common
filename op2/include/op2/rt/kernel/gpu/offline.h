#pragma once

// The calls on what the application compiled, made from op2_kernels with the
// runtime that registered it: an offline kernel's attributes and launches,
// and the device addresses of its __constant__ symbols.  The generated code
// registers offline(kernel) as a strategy's offline function and symbol(...)
// as a parameter's device copy (op2/rt/gpu/loop.h).

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/gpu/loop.h>

namespace op::rt::kernel::gpu {

namespace detail {

inline op::rt::abi::KernelFigures figures(const void *function) {
    gpuFuncAttributes_t attributes{};
    GPU_CHECK(gpuFuncGetAttributes(&attributes, function));
    return {attributes.numRegs, static_cast<int>(attributes.localSizeBytes),
            static_cast<int>(attributes.sharedSizeBytes), attributes.maxThreadsPerBlock};
}

inline void allow_shared(const void *function, int bytes) {
    GPU_CHECK(gpuFuncSetAttribute(function, gpuFuncAttributeMaxDynamicSharedMemorySize,
                                  bytes));
}

inline void launch(const void *function, int blocks, int block_size, void **args,
                   int shared_bytes) {
    GPU_CHECK(gpuLaunchKernel(function, blocks, block_size, args, shared_bytes, 0));
    GPU_CHECK(gpuPeekAtLastError());
}

inline constexpr op::rt::gpu::OfflineCalls calls{&figures, &allow_shared, &launch};

} // namespace detail

// A kernel of the application's, as a strategy's offline function.
inline op::rt::gpu::OfflineFunction offline(const void *function) {
    return {function, &detail::calls};
}

// The device address of a __constant__ symbol of the application's.
inline void *symbol(const void *symbol) {
    void *address = nullptr;
    GPU_CHECK(gpuGetSymbolAddress(&address, symbol));
    return address;
}

} // namespace op::rt::kernel::gpu
