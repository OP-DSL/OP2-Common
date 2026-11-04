#pragma once

// The GPU's chunk plans, built from the plan building blocks under the
// device's limits: a chunk is a block's elements, its staging fits the
// shared memory a block may use and the 16-bit stage word it encodes its
// slots as, and a thread colour fits a byte.  Host code with no device calls; DevicePlan
// (op2/rt/gpu/chunk_encode.h) uploads a plan as the kernels read it.  The
// planners are defined in src/rt/gpu/op_gpu_chunk_plan.cpp.

#include <op2/rt/abi/staging.h>
#include <op2/rt/plan/chunks.h>
#include <op2/rt/plan/staging.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace op::rt::gpu {

// What a chunk plan is built for: a chunk is a block's elements, its
// staging fits the shared memory a block may use, and its per-element arrays
// have the device's stride (op2/rt/gpu/layout.h).
struct ChunkSpec {
    int block_size = 0;
    std::size_t shared_bytes = 0;
    int stride = 0;
};

// Each launch runs a batch of chunks, a block per chunk, each chunk
// accumulating into its targets' slots in shared memory, its slots as the
// kernels read them: a 16-bit stage word each, unstaged_word for a
// reference left in global memory.
struct ChunkPlan : plan::ChunkPlan<abi::StageWord> {
    // chunk_colouring: per element its thread colour within its chunk, per
    // chunk its colour count.
    std::vector<std::uint8_t> thread_colours;
    std::vector<int> chunk_thread_colours;
    int max_thread_colours = 0;

    // A wavefront plan runs a section as one batch, each position after its
    // predecessors.
    bool wavefront = false;
    std::vector<int> wave_pred_offsets;
    std::vector<int> wave_preds;

    // Per batch, the shared memory its largest chunk needs, and the most
    // chunks a batch launches.
    std::vector<std::size_t> batch_scratch_bytes;
    int max_batch = 0;

    std::size_t references = 0;
    std::size_t distinct_targets = 0;

    std::size_t scratch_bytes(int section, int batch) const {
        return batch_scratch_bytes[
            static_cast<std::size_t>(batches.batch_index(section, batch))];
    }

    // Once uploaded, drop what only the device reads; the batches, their
    // scratch, the wavefront flag and the figures stay.
    void release_uploaded() {
        plan::Staging kept;
        kept.stride = staging.stride;
        kept.unstaged_singles = staging.unstaged_singles;
        staging = std::move(kept);
        stage_words = {};
        thread_colours = {};
        chunk_thread_colours = {};
        wave_pred_offsets = {};
        wave_preds = {};
    }
};

// A rejection's reason is one of plan::stage_chunks', low_compression or
// too_many_colours.
using ChunkResult = plan::PlanResult<ChunkPlan>;

// chunk_atomics: a section's chunks run at once and may share targets, so
// every slot flushes with a global atomic.  A plan that barely combines
// references is rejected, unless its single references stay unstaged: then
// it costs little more than global atomics, since those are its singles.
ChunkResult plan_chunk_atomics(std::span<const op_arg> args,
                               const op::rt::loop::Sections& sections,
                               const abi::ArgGroups& groups, const ChunkSpec& spec,
                               bool unstaged_singles, double min_compression);

// chunk_colouring: the chunks of a batch share no target, and a chunk's
// elements of one thread colour neither, so every slot is seeded and stored
// with plain loads and stores.  A wavefront plan colours window chunks at a
// time (0: a section at once) and runs each section as one batch.
ChunkResult plan_chunk_colouring(std::span<const op_arg> args,
                                 const op::rt::loop::Sections& sections,
                                 const abi::ArgGroups& groups, const ChunkSpec& spec,
                                 bool wavefront, int window);

} // namespace op::rt::gpu
