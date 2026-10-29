#pragma once

// The block a GPU kernel runs as.  Device code.

#include <op2/rt/kernel/prelude.h>

// Declares blockDim under offline HIP.
#if defined(__HIPCC__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

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

} // namespace op::rt::kernel::gpu
