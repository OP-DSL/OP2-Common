#pragma once

// The device helpers the GPU skeletons and their hooks share.  Device code.

#include <op2/rt/kernel/prelude.h>

namespace op::rt::kernel::gpu {

// The block's thread count, for a function a kernel inlines.  HIP's
// blockDim.x allows for a smaller last block, which a kernel's uniform block
// size folds away only in code compiled as the kernel's own, and a function
// is simplified before it is inlined; HIP launches whole blocks, so read the
// block size the launch gave.  Unsigned, as blockDim.x is: a signed stride
// lets the compiler widen the loops that add it.
__device__ __forceinline__ unsigned block_size() {
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_workgroup_size_x();
#else
    return blockDim.x;
#endif
}

// Return x unchanged, but opaque to the optimiser.
__device__ __forceinline__ int opaque(int x) {
#if defined(__CUDA_ARCH__)
    asm volatile("" : "+r"(x));
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(x));
#endif
    return x;
}

} // namespace op::rt::kernel::gpu
