#pragma once

// How a GPU kernel has its parameters: a JIT kernel's baked in, or given at
// run time in place of their baked values, and a small array by value as a
// kernel argument.  Compiled on the host and for the device, by NVRTC and
// hipRTC too.

#include <op2/rt/kernel/prelude.h>

namespace op::rt::kernel::gpu {

// A small array passed by value, as a kernel argument.
template<typename T, unsigned N>
struct Array {
    T data[N];
};

// A JIT kernel's parameter given at run time: defined as one of these in
// place of its baked value, so param_value(name, argument) reads the
// kernel's argument instead.  A baked array comes as its pointer.
struct RuntimeParam {};

template<typename T>
OP_KERNEL_DEVICE constexpr const T& param_value(const T& baked, const T&) {
    return baked;
}

template<typename T>
OP_KERNEL_DEVICE constexpr const T& param_value(RuntimeParam, const T& argument) {
    return argument;
}

template<typename T, unsigned N>
OP_KERNEL_DEVICE constexpr const T *param_value(const T (&baked)[N], const T *) {
    return baked;
}

} // namespace op::rt::kernel::gpu
