#pragma once

// The calls on what the application compiled, made from op2_kernels with the
// runtime that registered it: an offline kernel's attributes and launches,
// and the device addresses of its __constant__ symbols.  The generated code
// describes calls as its loops' (op2/rt/abi/gpu/descriptor.h).

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/gpu/descriptor.h>

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

inline void *symbol(const void *symbol) {
    void *address = nullptr;
    GPU_CHECK(gpuGetSymbolAddress(&address, symbol));
    return address;
}

} // namespace detail

inline constexpr op::rt::abi::gpu::OfflineCalls calls{&detail::figures, &detail::allow_shared,
                                                      &detail::launch, &detail::symbol};

} // namespace op::rt::kernel::gpu
