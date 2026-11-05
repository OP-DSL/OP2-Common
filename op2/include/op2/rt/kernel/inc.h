#pragma once

// The increments a translated kernel applies, on the host and on the device:
// in place, or atomically, and the hooks a kernel is instantiated with.

#include <op2/rt/kernel/prelude.h>

#if !defined(OP_KERNEL_GPU)
#include <atomic>
#endif

namespace op::rt::kernel {

// The plain form keeps Fortran's x = x + e conversion; the atomic one
// converts e first, as atomicAdd's operand type requires.
template<typename T, typename U>
OP_KERNEL_DEVICE inline void plain_inc(T& target, U value) {
    target = static_cast<T>(target + value);
}

#if !defined(OP_KERNEL_GPU)
// Host threads: a compare-and-swap loop for a floating-point target, a
// locked add for an integer one.
template<typename T, typename U>
inline void atomic_inc(T& target, U value) {
    std::atomic_ref<T>{target}.fetch_add(static_cast<T>(value),
                                         std::memory_order_relaxed);
}
#else
template<typename T, typename U>
OP_KERNEL_DEVICE inline void atomic_inc(T& target, U value) {
#if defined(__HIP_DEVICE_COMPILE__)
    // Name the address space: a flat atomic, as a target that is either
    // staged or global gets, is also guarded against scratch.
    if (__builtin_amdgcn_is_shared(&target))
        __hip_atomic_fetch_add((__attribute__((address_space(3))) T *)&target,
                               static_cast<T>(value), __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_WORKGROUP);
    else
        __hip_atomic_fetch_add((__attribute__((address_space(1))) T *)&target,
                               static_cast<T>(value), __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
#else
    // CUDA's atomicAdd has no signed 64-bit form: add the two's-complement bits.
    if constexpr (sizeof(T) == 8 && is_integer_type<T>::value)
        atomicAdd(reinterpret_cast<unsigned long long *>(&target),
                  static_cast<unsigned long long>(static_cast<T>(value)));
    else
        atomicAdd(&target, static_cast<T>(value));
#endif
}

#endif

// A translated kernel that takes an indirect OP_INC argument is a template over
// its increment hook, op2_inc, and applies each increment of it as
// op2_inc::add(target, value): in place where no other thread updates the
// target meanwhile, atomically where one may.
struct PlainInc {
    template<typename T, typename U>
    OP_KERNEL_DEVICE static void add(T& target, U value) { plain_inc(target, value); }
};

struct AtomicInc {
    template<typename T, typename U>
    OP_KERNEL_DEVICE static void add(T& target, U value) { atomic_inc(target, value); }
};

} // namespace op::rt::kernel
