#pragma once

// The increment hooks a translated kernel's op2_inc is defined as, on the
// host and on the device: in place, or atomically.

#include <op2/rt/kernel/prelude.h>

// Declares atomicAdd under offline HIP.
#if defined(__HIPCC__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <atomic>
#endif

namespace op::rt::kernel {

// A translated kernel applies each indirect increment through op2_inc, which
// every strategy's generated code defines as one of these.  The plain form
// keeps Fortran's x = x + e conversion; the atomic one converts e first, as
// atomicAdd's operand type requires.
template<typename T, typename U>
OP_KERNEL_DEVICE inline void plain_inc(T& target, U value) {
    target = static_cast<T>(target + value);
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)
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

} // namespace op::rt::kernel
