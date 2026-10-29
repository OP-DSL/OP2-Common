#pragma once

// The CUDA and HIP data layout the kernels index by, as the CUDA data layer
// lays out the device's maps and SoA dats.  Host code.

#include <op_lib_core.h>

namespace op::rt::gpu {

// The stride of a loop's per-element device arrays: the set's elements, and
// for an indirect loop its exec halo too, rounded up to a warp.
inline int element_stride(op_set set, bool direct) {
    int elements = direct ? set->size : set->size + set->exec_size;
    return (elements + 31) & ~31;
}

} // namespace op::rt::gpu
