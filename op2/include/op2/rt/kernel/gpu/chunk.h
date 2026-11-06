#pragma once

// The chunked strategies' kernels: a block takes a batch's chunks in turn
// (op2/rt/abi/gpu/launch.h), stages the chunk's targets in shared memory,
// applies its elements to them, then writes each target once, with the
// barriers between.  Device code; the generated wrapper passes the number of
// staged regions, its record's fields (OP_ABI_GPU_ARG) and its hooks:
// stage(counts, slots) lays out the chunk's regions, counts[r] slots each,
// and under chunk_colouring seeds them from region r's slot list at
// slots[r]; element(n) applies element n; and write(slots) writes the slots
// back, flushing them with global atomics under chunk_atomics and storing
// them under chunk_colouring.  The skeleton derives a chunk's counts and
// slots from its own parameters as it starts the chunk: a hook reading the
// record's arrays itself has them, under HIP, as generic pointers, and slots
// derived at the write cost CUDA registers.

#include <op2/rt/abi/gpu/launch.h>
#include <op2/rt/kernel/gpu/device.h>
#include <op2/rt/kernel/gpu/wavefront.h>
#include <op2/rt/kernel/prelude.h>

namespace op::rt::kernel::gpu {

// One chunk: stage its targets, run body over its elements, then write each
// target once.
template<typename Stage, typename Body, typename Write>
__device__ __forceinline__ void with_chunk(int chunk, int regions,
                                           const int *__restrict stage_counts,
                                           const int *__restrict slot_offsets, Stage&& stage,
                                           Body&& body, Write&& write) {
    const int *counts = stage_counts + chunk * regions;
    const int *slots = slot_offsets + chunk * regions;
    stage(counts, slots);
    __syncthreads();

    body();
    __syncthreads();

    // One write per distinct target reaches global memory.
    write(slots);

    // Keep the next chunk from reusing memory a warp is still writing.
    __syncthreads();
}

// chunk_atomics: a batch's chunks may share targets, so a chunk accumulates
// into its cleared slots with shared atomics and flushes them with global
// ones.
template<typename Stage, typename Element, typename Write>
__device__ __forceinline__ void chunk_atomics(int regions,
                                              OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_PARAM)
                                              Stage&& stage, Element&& element,
                                              Write&& write) {
    for (int chunk = chunk_begin + blockIdx.x; chunk < chunk_end; chunk += gridDim.x) {
        const int begin = chunk_offsets[chunk];
        const int end = chunk_offsets[chunk + 1];
        with_chunk(chunk, regions, stage_counts, slot_offsets, stage, [&] {
            for (int n = begin + threadIdx.x; n < end; n += block_size())
                element(n);
        }, write);
    }
}

// The chunks of a chunk_colouring batch, positions [chunk_begin, chunk_end)
// through chunk_order, which share no target: grid-stride, or under a
// wavefront taken in turn from the section's counter, each once the earlier
// positions sharing one of its targets are done.  visit(chunk, begin, end)
// runs one, and signals it done under a wavefront.
template<bool wavefront, typename Visit>
__device__ __forceinline__ void colour_chunks(OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_PARAM)
                                              Visit&& visit) {
    __shared__ int taken;
    for (int i = wavefront ? wave_take(wave_ticket, chunk_begin, taken) : chunk_begin + blockIdx.x;
         i < chunk_end;
         i = wavefront ? wave_take(wave_ticket, chunk_begin, taken) : i + gridDim.x) {
        if (wavefront)
            wave_wait(wave_pred_offsets, wave_preds, wave_done, i, wave_epoch);
        const int chunk = chunk_order[i];
        visit(chunk, chunk_offsets[chunk], chunk_offsets[chunk + 1]);
        if (wavefront)
            wave_signal(wave_done, i, wave_epoch);
    }
}

// chunk_colouring: every slot is seeded from its target and stored back,
// and a chunk's elements apply one thread colour at a time, so slots are
// read and written with plain loads and stores.
template<bool wavefront, typename Stage, typename Element, typename Write>
__device__ __forceinline__ void chunk_colouring(int regions,
                                                OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_PARAM)
                                                Stage&& stage, Element&& element,
                                                Write&& write) {
    auto visit = [&](int chunk, int begin, int end) {
        with_chunk(chunk, regions, stage_counts, slot_offsets, stage, [&] {
            const int colours = chunk_thread_colours[chunk];
            for (int base = begin; base < end; base += block_size()) {
                const int n = base + threadIdx.x;
                const int colour = n < end ? thread_colours[n] : -1;
                for (int pass = 0; pass < colours; ++pass) {
                    // Opaque, so this element's address arithmetic is not
                    // hoisted out of the pass loop and kept live through it.
                    if (colour == pass)
                        element(opaque(n));
                    __syncthreads();
                }
            }
        }, write);
    };
    colour_chunks<wavefront>(OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_ARG) visit);
}

// chunk_colouring when every element of a chunk can compute at once: its
// groups only increment and read, so each thread computes its element's
// increments into its own Locals (compute(n, locals)), and only applying
// them to the slots (apply(n, locals)) goes one thread colour at a time.
template<bool wavefront, typename Locals, typename Stage, typename Compute, typename Apply,
         typename Write>
__device__ __forceinline__ void chunk_colouring_parallel(int regions,
                                                         OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_PARAM)
                                                         Stage&& stage, Compute&& compute,
                                                         Apply&& apply, Write&& write) {
    auto visit = [&](int chunk, int begin, int end) {
        with_chunk(chunk, regions, stage_counts, slot_offsets, stage, [&] {
            const int colours = chunk_thread_colours[chunk];
            for (int base = begin; base < end; base += block_size()) {
                const int n = base + threadIdx.x;
                const int colour = n < end ? thread_colours[n] : -1;
                Locals locals{};
                if (colour >= 0)
                    compute(n, locals);

                // Every staged read completes before any slot changes.
                __syncthreads();

                for (int pass = 0; pass < colours; ++pass) {
                    if (colour == pass)
                        apply(n, locals);
                    __syncthreads();
                }
            }
        }, write);
    };
    colour_chunks<wavefront>(OP_ABI_GPU_CHUNK_FIELDS(OP_ABI_GPU_ARG) visit);
}

} // namespace op::rt::kernel::gpu
