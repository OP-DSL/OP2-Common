#include <op2/rt/cpu/chunk_plan.h>

#include <algorithm>
#include <memory>
#include <utility>

namespace op::rt::cpu {

ChunkResult plan_chunk_atomics(std::span<const op_arg> args,
                               const op::rt::loop::Sections& sections,
                               const plan::ArgGroups& groups, int size,
                               double min_compression) {
    plan::StagedChunks staged;
    auto reason = plan::stage_chunks(args, sections, groups, false, size, 1,
                                     chunk_scratch_limit, false, staged);
    if (!reason.empty())
        return {reason, nullptr};
    double compression = staged.targets.compression();
    if (compression < min_compression)
        return {"low_compression", nullptr, compression};

    auto out = std::make_unique<ChunkPlan>();
    out->chunks = std::move(staged.chunks);
    out->staging = std::move(staged.staging);
    out->batches = plan::section_batches(out->chunks);
    for (auto bytes : out->staging.chunk_scratch_bytes)
        out->max_scratch_bytes = std::max(out->max_scratch_bytes, bytes);
    out->compression = compression;
    return {{}, std::move(out)};
}

ChunkResult plan_chunk_colouring(std::span<const op_arg> args,
                                 const op::rt::loop::Sections& sections, int size) {
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

} // namespace op::rt::cpu
