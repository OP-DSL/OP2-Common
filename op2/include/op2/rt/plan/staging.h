#pragma once

// The staged kernels' scratch, the layout every backend's staged kernels
// read (op2/rt/abi/staging.h): a slot per distinct target of each chunk's
// region, numbered in target order, and a stage word per staged reference
// and element naming its slot.  stage_chunks runs the whole pipeline under a
// backend's lanes and scratch budget.

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/staging.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/plan/chunks.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace op::rt::plan {

struct Staging {
    // Per reference and element, stage_words[reference * stride + element].
    int stride = 0;
    std::vector<abi::StageWord> stage_words;
    // Per region its first slot in slot_targets, then the slot count.
    std::vector<int> slot_offsets;
    std::vector<int> slot_targets;
    // Per chunk, the scratch bytes its regions need.
    std::vector<std::size_t> chunk_scratch_bytes;
    // References left in global memory, each its chunk's only one to its
    // target.
    std::size_t unstaged_singles = 0;
};

// A call's chunks for a staged kernel: the groups' references, the chunks,
// the targets each reaches, and their staging.
struct StagedChunks {
    References references;
    Chunks chunks;
    ChunkTargets targets;
    Staging staging;
};

// Chunk the sections at the largest multiple of lanes up to size, itself
// one, whose staging fits scratch_limit and the stage word, and stage them,
// each active reference of a chunk to a slot of its region (unstaged_singles
// leaves a target the chunk references once in global memory).  Or why not:
// no_active_argument; incompatible_argument when two groups reach one dat,
// or an argument outside the groups reaches a grouped dat or makes an update
// the groups would not order (an indirect increment, or with read_write an
// indirect read-write); insufficient_scratch.
std::string_view stage_chunks(std::span<const op_arg> args,
                              const op::rt::loop::Sections& sections,
                              const abi::ArgGroups& groups, bool read_write, int size,
                              int lanes, std::size_t scratch_limit,
                              bool unstaged_singles, StagedChunks& out);

} // namespace op::rt::plan
