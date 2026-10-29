#pragma once

// The staged kernels' scratch: a slot per distinct target of each chunk's
// region, numbered in target order, and per staged reference and element
// the slot it names.  stage_chunks runs the whole pipeline under a backend's
// lanes, scratch budget and slot limit; the backend encodes the slots as its
// kernels read them (stage_words).

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/plan/chunks.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace op::rt::plan {

// The slot of a reference left in global memory: its chunk's only one to its
// target.
constexpr int unstaged_slot = -1;

struct Staging {
    // Per reference and element, slots[reference * stride + element]: the
    // slot the element's reference names in its chunk's region.
    int stride = 0;
    std::vector<int> slots;
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

// How a backend stages a call's chunks: up to size elements a chunk, a
// multiple of lanes; within scratch_limit bytes and max_slots slots a
// region, as its encoding allows; the slots stride apart, the backend's
// per-element arrays' stride; with read_write the groups also order indirect
// read-writes; under unstaged_singles a target a chunk references once stays
// in global memory.
struct StagingSpec {
    int size = 0;
    int lanes = 1;
    std::size_t scratch_limit = 0;
    int max_slots = 0;
    int stride = 0;
    bool read_write = false;
    bool unstaged_singles = false;
};

// Chunk the sections at the largest multiple of lanes up to size, itself
// one, whose staging fits the scratch and slot limits, and stage them, each
// active reference of a chunk to a slot of its region.  Or why
// not: no_active_argument; incompatible_argument when two groups reach one
// dat, or an argument outside the groups reaches a grouped dat or makes an
// update the groups would not order (an indirect increment, or with
// read_write an indirect read-write); insufficient_scratch.
std::string_view stage_chunks(std::span<const op_arg> args,
                              const op::rt::loop::Sections& sections,
                              const abi::ArgGroups& groups, const StagingSpec& spec,
                              StagedChunks& out);

// The slots as a backend's stage words: each a Word, an unstaged one as
// unstaged.
template<typename Word>
std::vector<Word> stage_words(std::span<const int> slots, Word unstaged) {
    std::vector<Word> words(slots.size());
    for (std::size_t i = 0; i < slots.size(); ++i)
        words[i] = slots[i] == unstaged_slot ? unstaged : static_cast<Word>(slots[i]);
    return words;
}

} // namespace op::rt::plan
