#pragma once

// The staged kernels' scratch, the layout every backend's staged kernels
// read (plan/staging_layout.h): a slot per distinct target of each chunk's
// region, numbered in target order, and a stage word per staged reference
// and element naming its slot.  stage_chunks runs the whole pipeline under a
// backend's lanes and scratch budget.

#include <plan/chunks.h>
#include <plan/staging_layout.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace op::plan {

struct Staging {
    // Per reference and element, stage_words[reference * stride + element].
    int stride = 0;
    std::vector<StageWord> stage_words;
    // Per region its first slot in slot_targets, then the slot count.
    std::vector<int> slot_offsets;
    std::vector<int> slot_targets;
    // Per chunk, the scratch bytes its regions need.
    std::vector<std::size_t> chunk_scratch_bytes;
    // References left in global memory, each its chunk's only one to its
    // target.
    std::size_t unstaged_singles = 0;
};

// The scratch bytes of a chunk with counts slots per group: the regions in
// group order, each aligned to its scalar size.
std::size_t scratch_bytes(const References& references, std::span<const int> counts);

// The largest multiple of lanes, up to size rounded up to lanes, whose
// chunks' worst-case staging fits scratch_bytes and the stage word; 0 if
// none does.
int staged_chunk_size(const References& references, int size, int lanes,
                      std::size_t scratch_bytes);

// Stage every active reference of each chunk.  unstaged_singles leaves a
// target the chunk references once in global memory.
Staging stage(const Chunks& chunks, const ChunkTargets& targets,
              const References& references, int stride, bool unstaged_singles);

// A call's chunks for a staged kernel: the groups' references, the chunks,
// the targets each reaches, and their staging.
struct StagedChunks {
    References references;
    Chunks chunks;
    ChunkTargets targets;
    Staging staging;
};

// Chunk the sections at the largest multiple of lanes up to size whose
// staging fits scratch_bytes, and stage them; or why not: a reason of
// grouped_references, or insufficient_scratch.
std::string_view stage_chunks(std::span<const op_arg> args,
                              const op::loop::Sections& sections,
                              const ArgGroups& groups, bool read_write, int size,
                              int lanes, std::size_t scratch_bytes,
                              bool unstaged_singles, StagedChunks& out);

} // namespace op::plan
