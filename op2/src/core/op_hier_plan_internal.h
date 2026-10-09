#pragma once

// Planner support shared by the hierarchical strategies' plan builders.

#include <op_hier_plan.h>

#include <cstddef>
#include <span>
#include <vector>

namespace op::f2c::detail {

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
    int requested_chunk_size = 0;
    std::span<const op_arg> args;
    std::span<const HierArgDescriptor> arg_descriptors;
    std::vector<ResolvedDat> dats;
};

// Resolve generated metadata and runtime identities into canonical plan input.
// Increments-only groups additionally assert every active argument is OP_INC.
HierFallbackReason resolve_input(
    op_set set, std::span<const op_arg> args, const HierArgGroups& groups,
    const HierPlanOptions& options, bool increments_only,
    ResolvedInput& resolved);

// Resolve the global target an argument reaches from one source element.
inline int arg_target(const op_arg& arg, int source) {
    return arg.map_data[static_cast<std::size_t>(source) *
                            static_cast<std::size_t>(arg.map->dim) +
                        static_cast<std::size_t>(arg.idx)];
}

} // namespace op::f2c::detail
