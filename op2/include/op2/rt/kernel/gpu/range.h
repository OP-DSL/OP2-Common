#pragma once

// The plain, atomics and colouring strategies' kernel: a grid-stride walk of
// a launch's range of elements (op2/rt/abi/gpu/launch.h), the loop's element
// function for each, through order under a set colouring.  Device code; the
// generated wrapper passes its record's fields and its element function.

#include <op2/rt/kernel/gpu/device.h>
#include <op2/rt/kernel/prelude.h>

namespace op::rt::kernel::gpu {

template<bool ordered, typename Element>
__device__ __forceinline__ void range(int start, int end, const int *__restrict order,
                                      Element&& element) {
    for (int i = threadIdx.x + blockIdx.x * block_size() + start; i < end;
         i += block_size() * gridDim.x)
        element(ordered ? order[i] : i);
}

} // namespace op::rt::kernel::gpu
