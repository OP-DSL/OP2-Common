#include <op2/rt/gpu/chunk_plan.h>

#include <algorithm>
#include <memory>
#include <utility>

namespace op::rt::gpu {

namespace {

// A plan holding the staged chunks and their figures; the references and
// targets stay in staged for the batching.
std::unique_ptr<ChunkPlan> staged_plan(plan::StagedChunks& staged) {
    auto out = std::make_unique<ChunkPlan>();
    plan::take_staged(staged, abi::unstaged_word, *out);
    out->references = staged.targets.total_references;
    out->distinct_targets = staged.targets.targets.size();
    return out;
}

// Size every batch's shared memory by its largest chunk.
void size_batches(ChunkPlan& out) {
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

} // namespace

ChunkResult plan_chunk_atomics(std::span<const op_arg> args,
                               const op::rt::loop::Sections& sections,
                               const abi::ArgGroups& groups, const ChunkSpec& spec,
                               bool unstaged_singles, double min_compression) {
    plan::StagedChunks staged;
    auto reason = plan::stage_chunks(args, sections, groups,
                                     {.size = spec.block_size,
                                      .lanes = spec.block_size,
                                      .scratch_limit = spec.shared_bytes,
                                      .max_slots = abi::max_slots,
                                      .stride = spec.stride,
                                      .unstaged_singles = unstaged_singles},
                                     staged);
    if (!reason.empty())
        return {reason, nullptr};
    auto out = staged_plan(staged);
    if (!unstaged_singles && out->compression < min_compression)
        return {"low_compression", nullptr, out->compression};

    out->batches = plan::section_batches(out->chunks);
    size_batches(*out);
    return {{}, std::move(out)};
}

ChunkResult plan_chunk_colouring(std::span<const op_arg> args,
                                 const op::rt::loop::Sections& sections,
                                 const abi::ArgGroups& groups, const ChunkSpec& spec,
                                 bool wavefront, int window) {
    plan::StagedChunks staged;
    auto reason = plan::stage_chunks(args, sections, groups,
                                     {.size = spec.block_size,
                                      .lanes = spec.block_size,
                                      .scratch_limit = spec.shared_bytes,
                                      .max_slots = abi::max_slots,
                                      .stride = spec.stride,
                                      .read_write = true},
                                     staged);
    if (!reason.empty())
        return {reason, nullptr};
    auto out = staged_plan(staged);

    auto colours = plan::colour_elements(out->chunks, staged.references);
    if (colours.max > 255)
        return {"too_many_colours", nullptr};
    out->thread_colours.assign(static_cast<std::size_t>(spec.stride), 0);
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
    size_batches(*out);
    return {{}, std::move(out)};
}

} // namespace op::rt::gpu
