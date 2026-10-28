#pragma once

// What a strategy's kernel compiles against beyond the translated kernel's
// own prelude: the increment hooks, the scratch staging helpers of the
// chunked strategies and the wavefront primitives.  A library any wrapper
// includes, generated or synthesised.

#include <kernel/prelude.h>
#include <plan/staging_layout.h>

// Declares atomicAdd for the staging helpers under offline HIP.
#if defined(__HIPCC__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <atomic>
#endif

namespace op::kernel {

/* Scratch staging helpers.

   A staged kernel carves one dense region per staged dat out of its dynamic
   shared allocation, accumulates into it, then writes each slot back to its
   target.  A region keeps each slot's components together,
   region[slot * dim + c], so with a JIT kernel's constant dim a component's
   offset is an immediate rather than a register.

   The AoS/SoA distinction never reaches them: a caller passes the global
   element as a Ptr, whose stride already encodes the layout (1 for AoS, the
   dat's SoA stride otherwise), exactly as the accumulation path does.

   Thread indices are parameters rather than reads of threadIdx, which keeps
   the staging helpers compilable for the host. */

// Carve the next region out of the shared buffer, advancing the cursor
// (alignof == sizeof for every supported scalar type).  Pad the pointer, not
// an integer round trip of it: that hides that it points into shared memory,
// so every increment through it becomes a generic atomic.  An empty region
// takes no padding, as the host sizes none for it.
template<typename T>
DEVICE inline T *stage_region(char *&cursor, IndexType count, IndexType dim) {
    if (count == 0)
        return (T *)cursor;
    constexpr size_t alignment = sizeof(T);
    cursor += (alignment - ((size_t)cursor & (alignment - 1))) &
              (alignment - 1);

    T *base = (T *)cursor;
    cursor += (size_t)count * (size_t)dim * sizeof(T);
    return base;
}

// Cooperatively zero a region before accumulating into it.
template<typename T>
DEVICE inline void stage_clear(T *region, IndexType elements, IndexType lane,
                               IndexType lanes) {
    for (IndexType i = lane; i < elements; i += lanes)
        region[i] = 0;
}

// Seed a slot from its target's current value.
template<typename T, typename Source>
DEVICE inline void stage_seed(Ptr<Source> target, T *region, IndexType slot,
                              IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        region[slot * dim + c] = target.data[c * target.stride];
}

// Add one element's per-thread increments into its staged slot.
template<typename T>
DEVICE inline void stage_add(Ptr<T> slot, const T *increment, IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        slot.data[c * slot.stride] += increment[c];
}

// Store a slot back to its target, which no other chunk running alongside
// this one reaches.
template<typename T>
DEVICE inline void stage_store(Ptr<T> target, const T *region, IndexType slot,
                               IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        target.data[c * target.stride] = region[slot * dim + c];
}

// A translated kernel applies each indirect increment through op2_inc, which
// every strategy's generated code defines as one of these.  The plain form
// keeps Fortran's x = x + e conversion; the atomic one converts e first, as
// atomicAdd's operand type requires.
template<typename T, typename U>
DEVICE inline void plain_inc(T& target, U value) {
    target = static_cast<T>(target + value);
}

// Return x unchanged, but opaque to the optimiser.
DEVICE inline int opaque(int x) {
#if defined(__CUDA_ARCH__)
    asm volatile("" : "+r"(x));
#elif defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(x));
#endif
    return x;
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)
// Host threads: a compare-and-swap loop for a floating-point target, a
// locked add for an integer one.
template<typename T, typename U>
inline void atomic_inc(T& target, U value) {
    std::atomic_ref<T>{target}.fetch_add(static_cast<T>(value),
                                         std::memory_order_relaxed);
}

template<typename T>
inline void stage_flush(Ptr<T> target, const T *region, IndexType slot,
                        IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        atomic_inc(target.data[c * target.stride], region[slot * dim + c]);
}
#else
template<typename T, typename U>
DEVICE inline void atomic_inc(T& target, U value) {
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
    atomicAdd(&target, static_cast<T>(value));
#endif
}

// Add a slot's accumulated increments into its target, which other chunks
// of the section may be adding to as well.
template<typename T>
DEVICE inline void stage_flush(Ptr<T> target, const T *region, IndexType slot,
                               IndexType dim) {
    for (IndexType c = 0; c < dim; ++c)
        atomicAdd(&target.data[c * target.stride], region[slot * dim + c]);
}

/* Wavefront colouring (OP_CHUNK_COLOURING_WAVEFRONT): a block takes positions
   from its section's counter, waits until each predecessor's done flag holds
   the call's epoch, and sets its own once the chunk is stored.  The flag is
   set with a device-scope release and polled relaxed, with one device-scope
   acquire fence once it is seen, so a chunk's seeds see the stores of
   predecessors that ran on another multiprocessor (or XCD) while a poll
   invalidates no cache. */
DEVICE inline int wave_poll(const int *flag) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __hip_atomic_load(const_cast<int *>(flag), __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__)
    int value;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];"
                 : "=r"(value) : "l"(flag) : "memory");
    return value;
#else
    return *flag;
#endif
}

DEVICE inline void wave_acquire() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#elif defined(__CUDA_ARCH__)
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

DEVICE inline void wave_release(int *flag, int value) {
#if defined(__HIP_DEVICE_COMPILE__)
    __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__)
    asm volatile("st.release.gpu.global.b32 [%0], %1;"
                 :: "l"(flag), "r"(value) : "memory");
#else
    *flag = value;
#endif
}

DEVICE inline void wave_pause() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__)
    __nanosleep(64);
#endif
}

// Take the block's next position; every thread returns it.
DEVICE inline int wave_take(int *ticket, int begin, int& taken) {
    if (threadIdx.x == 0)
        taken = begin + atomicAdd(ticket, 1);
    __syncthreads();
    return taken;
}

// Wait until every predecessor of position is done in this call.
DEVICE inline void wave_wait(const int *pred_offsets, const int *preds,
                             const int *done, int position, int epoch) {
    bool waited = false;
    for (int p = pred_offsets[position] + (int)threadIdx.x;
         p < pred_offsets[position + 1]; p += blockDim.x) {
        while (wave_poll(done + preds[p]) < epoch)
            wave_pause();
        waited = true;
    }
    if (waited)
        wave_acquire();
    __syncthreads();
}

// Mark position done; call after the barrier that ends its stores, which
// orders every thread's stores before the release.
DEVICE inline void wave_signal(int *done, int position, int epoch) {
    if (threadIdx.x == 0)
        wave_release(done + position, epoch);
}
#endif

} // namespace op::kernel
