#pragma once

// Planner support shared by the chunk partition's two kinds.

#include <loop/chunk_partition.h>

#include <loop/greedy_colour.h>

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace op::chunk::detail {

using op::loop::detail::arg_target;
using op::loop::detail::greedy_colour;

struct ResolvedDat {
    op_dat dat = nullptr;
    int dimension = 0;
    int target_extent = 0;
    // Doubles as the region's alignment: every supported scalar type has
    // alignof == sizeof.
    std::size_t scalar_size = 0;
};

struct ResolvedInput {
    int set_stride = 0;
    int chunk_size = 0;
    std::span<const op_arg> args;
    std::span<const ArgGroup> groups;
    std::vector<ResolvedDat> dats;
};

// Resolve the groups and the runtime identities behind them into canonical
// input, or name why the configuration is rejected.  The groups must cover
// every active indirect increment, and with read_write every read-write as
// well, for the strategy to see every conflicting update.
std::string_view resolve_input(const op::loop::Configuration& configuration,
                               const ArgGroups& groups,
                               const Constraints& constraints,
                               bool read_write, ResolvedInput& resolved);

// The largest lanes multiple, up to the requested chunk size, whose
// worst-case staging fits the scratch, or 0 if none does.
int staged_chunk_size(const ResolvedInput& resolved,
                      const Constraints& constraints);

// The scratch bytes a chunk needs, given its slot count per dat group.
std::size_t chunk_scratch_bytes(const ResolvedInput& resolved,
                                std::span<const int> counts);

// Chunk each section and stage every grouped reference of each chunk: a slot
// per distinct target, numbered in target order.  unstaged_singles leaves a
// target the chunk references once in global memory.
void build_staging(const ResolvedInput& resolved,
                   const op::loop::Sections& sections, int chunk_size,
                   Partition& partition, bool unstaged_singles);

// Size every batch's scratch by its largest chunk once the batch offsets
// are final.
void size_batches(Partition& partition);

Result build_coloured(const op::loop::Configuration& configuration,
                      const op::loop::Sections& sections,
                      const ArgGroups& groups, const Constraints& constraints,
                      const Options& options);

} // namespace op::chunk::detail
