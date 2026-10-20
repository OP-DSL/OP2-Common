#pragma once

// The OpenMP backend's chunk plans, built from the plan building blocks: a
// thread runs a chunk on its own, so a chunk needs no lanes, and a staged
// chunk's scratch is bounded only so its slots stay in the caches.

#include <plan/chunks.h>
#include <plan/staging.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

namespace op::cpu {

// Elements per chunk unless OP_CHUNK_SIZES says otherwise.
constexpr int default_chunk_size = 256;

// The scratch a staged chunk may use.
constexpr std::size_t chunk_scratch_bytes = std::size_t{1} << 20;

// Each batch's chunks shared out over the threads.
struct ChunkPlan {
    plan::Chunks chunks;
    plan::ChunkBatches batches;
    // chunk_atomics: each chunk's slots in its thread's scratch, and the
    // most bytes a chunk needs.
    plan::Staging staging;
    std::size_t scratch_bytes = 0;
    double compression = 0.0;
};

// rejected names why the call does not suit the strategy: a reason of
// plan::grouped_references or plan::updated_references,
// insufficient_scratch, or low_compression; a rejection has no plan, and a
// low_compression one keeps its compression.
struct ChunkResult {
    std::string_view rejected;
    std::unique_ptr<ChunkPlan> plan;
    double compression = 0.0;

    explicit operator bool() const { return rejected.empty() && plan != nullptr; }
};

// chunk_atomics: a section's chunks run at once and may share targets, so a
// chunk accumulates into its slots and flushes them atomically.  A plan that
// barely combines references is rejected.
inline ChunkResult plan_chunk_atomics(std::span<const op_arg> args,
                                      const op::loop::Sections& sections,
                                      const plan::ArgGroups& groups, int size,
                                      double min_compression) {
    plan::References references;
    auto reason = plan::grouped_references(args, groups, false, references);
    if (!reason.empty())
        return {reason, nullptr};
    int chunk_size = plan::staged_chunk_size(references, size, 1, chunk_scratch_bytes);
    if (chunk_size == 0)
        return {"insufficient_scratch", nullptr};

    auto out = std::make_unique<ChunkPlan>();
    out->chunks = plan::chunk_sections(sections, chunk_size);
    auto targets = plan::chunk_targets(out->chunks, references);
    out->staging = plan::stage(out->chunks, targets, references,
                               sections.set_stride(), false);
    out->batches = plan::section_batches(out->chunks);
    for (auto bytes : out->staging.chunk_scratch_bytes)
        out->scratch_bytes = std::max(out->scratch_bytes, bytes);
    out->compression = targets.compression();
    if (out->compression < min_compression)
        return {"low_compression", nullptr, out->compression};
    return {{}, std::move(out)};
}

// chunk_colouring: the chunks of a batch share no target, so a chunk runs
// the plain body over its elements.
inline ChunkResult plan_chunk_colouring(std::span<const op_arg> args,
                                        const op::loop::Sections& sections,
                                        int size) {
    plan::References references;
    auto reason = plan::updated_references(args, references);
    if (!reason.empty())
        return {reason, nullptr};

    auto out = std::make_unique<ChunkPlan>();
    out->chunks = plan::chunk_sections(sections, size);
    auto targets = plan::chunk_targets(out->chunks, references);
    out->batches = plan::colour_chunks(out->chunks, targets);
    out->compression = targets.compression();
    return {{}, std::move(out)};
}

} // namespace op::cpu
