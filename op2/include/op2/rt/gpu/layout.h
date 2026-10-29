#pragma once

// The CUDA and HIP data layout the kernels index by, as the CUDA data layer
// lays out the device's maps and SoA dats: their per-element arrays padded
// to a warp.  Host code.

#include <op_lib_core.h>
#include <op_rt_support.h>

namespace op::rt::gpu {

// A per-element array's stride for elements: rounded up to a warp.
constexpr int pitch(int elements) { return (elements + 31) & ~31; }

// The stride of a loop's per-element device arrays: the set's elements, and
// for an indirect loop its exec halo too.
inline int element_stride(op_set set, bool direct) {
    return pitch(direct ? set->size : set->size + set->exec_size);
}

// The stride of a dat the device keeps as SoA, from an argument on it: its
// set's elements with the halos it holds.
inline int dat_stride(const op_arg& arg) {
    return pitch(getSetSizeFromOpArg(const_cast<op_arg *>(&arg)));
}

} // namespace op::rt::gpu
