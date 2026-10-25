#pragma once

// The GPU's chunk plans, built from the plan building blocks under the
// device's limits: a chunk is a multiple of the block and its staging fits
// the shared memory a block may use and the 16-bit stage word, and a thread
// colour fits a byte.  Host code with no device calls; DevicePlan
// (gpu/chunk_encode.h) uploads a plan as the kernels read it.

#include <plan/chunks.h>
#include <plan/staging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace op::gpu {

// What a chunk may use: chunks are multiples of the block, and a chunk's
// staging fits the shared memory.
struct ChunkLimits {
    int block_size = 0;
    std::size_t shared_bytes = 0;
};

// Each launch runs a batch of chunks, a block per chunk, each chunk
// accumulating into its targets' slots in shared memory.
struct ChunkPlan {
    plan::Chunks chunks;
    plan::Staging staging;
    plan::ChunkBatches batches;

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

    // Per batch, the shared memory its largest chunk needs.
    std::vector<std::size_t> batch_scratch_bytes;

    std::size_t references = 0;
    std::size_t distinct_targets = 0;
    double compression = 0.0;

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
        thread_colours = {};
        chunk_thread_colours = {};
        wave_pred_offsets = {};
        wave_preds = {};
    }
};

// A rejection's reason is one of plan::stage_chunks', low_compression or
// too_many_colours.
using ChunkResult = plan::PlanResult<ChunkPlan>;

namespace detail {

// A plan holding the staged chunks and their figures; the references and
// targets stay in staged for the batching.
inline std::unique_ptr<ChunkPlan> staged_plan(plan::StagedChunks& staged) {
    auto out = std::make_unique<ChunkPlan>();
    out->chunks = std::move(staged.chunks);
    out->staging = std::move(staged.staging);
    out->references = staged.targets.total_references;
    out->distinct_targets = staged.targets.targets.size();
    out->compression = staged.targets.compression();
    return out;
}

// Size every batch's shared memory by its largest chunk.
inline void size_batches(ChunkPlan& out) {
    out.batch_scratch_bytes.clear();
    for (int batch = 0; batch < out.batches.count(); ++batch) {
        std::size_t bytes = 0;
        for (int position = out.batches.batch_offsets[static_cast<std::size_t>(batch)];
             position < out.batches.batch_offsets[static_cast<std::size_t>(batch) + 1];
             ++position)
            bytes = std::max(bytes, out.staging.chunk_scratch_bytes[
                                        static_cast<std::size_t>(out.batches.chunk(position))]);
        out.batch_scratch_bytes.push_back(bytes);
    }
}

} // namespace detail

// chunk_atomics: a section's chunks run at once and may share targets, so
// every slot flushes with a global atomic.  A plan that barely combines
// references is rejected, unless its single references stay unstaged: then
// it costs little more than global atomics, since those are its singles.
inline ChunkResult plan_chunk_atomics(std::span<const op_arg> args,
                                      const op::loop::Sections& sections,
                                      const plan::ArgGroups& groups, int size,
                                      const ChunkLimits& limits, bool unstaged_singles,
                                      double min_compression) {
    plan::StagedChunks staged;
    auto reason = plan::stage_chunks(args, sections, groups, false, size, limits.block_size,
                                     limits.shared_bytes, unstaged_singles, staged);
    if (!reason.empty())
        return {reason, nullptr};
    auto out = detail::staged_plan(staged);
    if (!unstaged_singles && out->compression < min_compression)
        return {"low_compression", nullptr, out->compression};

    out->batches = plan::section_batches(out->chunks);
    detail::size_batches(*out);
    return {{}, std::move(out)};
}

// chunk_colouring: the chunks of a batch share no target, and a chunk's
// elements of one thread colour neither, so every slot is seeded and stored
// with plain loads and stores.  A wavefront plan colours window chunks at a
// time (0: a section at once) and runs each section as one batch.
inline ChunkResult plan_chunk_colouring(std::span<const op_arg> args,
                                        const op::loop::Sections& sections,
                                        const plan::ArgGroups& groups, int size,
                                        const ChunkLimits& limits, bool wavefront,
                                        int window) {
    plan::StagedChunks staged;
    auto reason = plan::stage_chunks(args, sections, groups, true, size, limits.block_size,
                                     limits.shared_bytes, false, staged);
    if (!reason.empty())
        return {reason, nullptr};
    auto out = detail::staged_plan(staged);

    auto colours = plan::colour_elements(out->chunks, staged.references);
    if (colours.max > 255)
        return {"too_many_colours", nullptr};
    out->thread_colours.assign(static_cast<std::size_t>(sections.set_stride()), 0);
    std::copy(colours.colours.begin(), colours.colours.end(), out->thread_colours.begin());
    out->chunk_thread_colours = std::move(colours.counts);
    out->max_thread_colours = colours.max;

    if (wavefront) {
        auto wave = plan::wavefront(
            plan::colour_chunks(out->chunks, staged.targets, window),
            staged.targets);
        out->batches = std::move(wave.batches);
        out->wavefront = true;
        out->wave_pred_offsets = std::move(wave.pred_offsets);
        out->wave_preds = std::move(wave.preds);
    } else {
        out->batches = plan::colour_chunks(out->chunks, staged.targets);
    }
    detail::size_batches(*out);
    return {{}, std::move(out)};
}

} // namespace op::gpu
