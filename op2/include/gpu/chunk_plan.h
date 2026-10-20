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

// Chunks per wavefront window when the run does not set one.
constexpr int default_wave_window = 4096;

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

    double compression() const {
        return distinct_targets > 0 ? static_cast<double>(references) /
                                          static_cast<double>(distinct_targets)
                                    : 0.0;
    }

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

// rejected names why the call does not suit the strategy: a reason of
// plan::grouped_references, insufficient_scratch, low_compression or
// too_many_colours; a rejection has no plan, and a low_compression one keeps
// its compression.
struct ChunkResult {
    std::string_view rejected;
    std::unique_ptr<ChunkPlan> plan;
    double compression = 0.0;

    explicit operator bool() const { return rejected.empty() && plan != nullptr; }
};

namespace detail {

// The chunks, their targets and staging common to both strategies.
inline std::string_view stage_chunks(std::span<const op_arg> args,
                                     const op::loop::Sections& sections,
                                     const plan::ArgGroups& groups, bool read_write,
                                     int size, const ChunkLimits& limits,
                                     bool unstaged_singles, plan::References& references,
                                     plan::ChunkTargets& targets, ChunkPlan& out) {
    auto reason = plan::grouped_references(args, groups, read_write, references);
    if (!reason.empty())
        return reason;
    int chunk_size = plan::staged_chunk_size(references, size, limits.block_size,
                                             limits.shared_bytes);
    if (chunk_size == 0)
        return "insufficient_scratch";

    out.chunks = plan::chunk_sections(sections, chunk_size);
    targets = plan::chunk_targets(out.chunks, references);
    out.staging = plan::stage(out.chunks, targets, references, sections.set_stride(),
                              unstaged_singles);
    out.references = targets.total_references;
    out.distinct_targets = targets.targets.size();
    return {};
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
    auto out = std::make_unique<ChunkPlan>();
    plan::References references;
    plan::ChunkTargets targets;
    auto reason = detail::stage_chunks(args, sections, groups, false, size, limits,
                                       unstaged_singles, references, targets, *out);
    if (!reason.empty())
        return {reason, nullptr};

    out->batches = plan::section_batches(out->chunks);
    detail::size_batches(*out);
    if (!unstaged_singles && out->compression() < min_compression)
        return {"low_compression", nullptr, out->compression()};
    return {{}, std::move(out)};
}

// chunk_colouring: the chunks of a batch share no target, and a chunk's
// elements of one thread colour neither, so every slot is seeded and stored
// with plain loads and stores.  A wavefront plan colours window chunks at a
// time and runs each section as one batch.
inline ChunkResult plan_chunk_colouring(std::span<const op_arg> args,
                                        const op::loop::Sections& sections,
                                        const plan::ArgGroups& groups, int size,
                                        const ChunkLimits& limits, bool wavefront,
                                        int window) {
    auto out = std::make_unique<ChunkPlan>();
    plan::References references;
    plan::ChunkTargets targets;
    auto reason = detail::stage_chunks(args, sections, groups, true, size, limits,
                                       false, references, targets, *out);
    if (!reason.empty())
        return {reason, nullptr};

    auto colours = plan::colour_elements(out->chunks, references);
    if (colours.max > 255)
        return {"too_many_colours", nullptr};
    out->thread_colours.assign(static_cast<std::size_t>(sections.set_stride()), 0);
    std::copy(colours.colours.begin(), colours.colours.end(), out->thread_colours.begin());
    out->chunk_thread_colours = std::move(colours.counts);
    out->max_thread_colours = colours.max;

    if (wavefront) {
        auto wave = plan::wavefront(
            plan::colour_chunks(out->chunks, targets,
                                window > 0 ? window : default_wave_window),
            targets);
        out->batches = std::move(wave.batches);
        out->wavefront = true;
        out->wave_pred_offsets = std::move(wave.pred_offsets);
        out->wave_preds = std::move(wave.preds);
    } else {
        out->batches = plan::colour_chunks(out->chunks, targets);
    }
    detail::size_batches(*out);
    return {{}, std::move(out)};
}

} // namespace op::gpu
