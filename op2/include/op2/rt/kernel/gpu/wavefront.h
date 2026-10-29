#pragma once

// The wavefront primitives of chunk_colouring's wavefront variant: device
// code only.

#include <op2/rt/kernel/gpu/block.h>
#include <op2/rt/kernel/prelude.h>

// Declares atomicAdd under offline HIP.
#if defined(__HIPCC__) && !defined(__HIPCC_RTC__)
#include <hip/hip_runtime.h>
#endif

namespace op::rt::kernel {

/* Wavefront colouring (OP_CHUNK_COLOURING_WAVEFRONT): a block takes positions
   from its section's counter, waits until each predecessor's done flag holds
   the call's epoch, and sets its own once the chunk is stored.  The flag is
   set with a device-scope release and polled relaxed, with one device-scope
   acquire fence once it is seen, so a chunk's seeds see the stores of
   predecessors that ran on another multiprocessor (or XCD) while a poll
   invalidates no cache. */
OP_KERNEL_DEVICE inline int wave_poll(const int *flag) {
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

OP_KERNEL_DEVICE inline void wave_acquire() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
#elif defined(__CUDA_ARCH__)
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

OP_KERNEL_DEVICE inline void wave_release(int *flag, int value) {
#if defined(__HIP_DEVICE_COMPILE__)
    __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#elif defined(__CUDA_ARCH__)
    asm volatile("st.release.gpu.global.b32 [%0], %1;"
                 :: "l"(flag), "r"(value) : "memory");
#else
    *flag = value;
#endif
}

OP_KERNEL_DEVICE inline void wave_pause() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__)
    __nanosleep(64);
#endif
}

// Take the block's next position; every thread returns it.
OP_KERNEL_DEVICE inline int wave_take(int *ticket, int begin, int& taken) {
    if (threadIdx.x == 0)
        taken = begin + atomicAdd(ticket, 1);
    __syncthreads();
    return taken;
}

// Wait until every predecessor of position is done in this call.
OP_KERNEL_DEVICE inline void wave_wait(const int *pred_offsets, const int *preds,
                             const int *done, int position, int epoch) {
    bool waited = false;
    for (int p = pred_offsets[position] + (int)threadIdx.x;
         p < pred_offsets[position + 1]; p += gpu::block_size()) {
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
OP_KERNEL_DEVICE inline void wave_signal(int *done, int position, int epoch) {
    if (threadIdx.x == 0)
        wave_release(done + position, epoch);
}

} // namespace op::rt::kernel
