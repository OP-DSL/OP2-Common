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
FallbackReason resolve_input(
    op_set set, std::span<const op_arg> args, const HierArgGroups& groups,
    const HierPlanOptions& options, bool increments_only,
    ResolvedInput& resolved);

// Which staged owners a plan marks exclusive, so their flush stores rather
// than adds atomically.
enum class ExclusiveOwners {
    none,     // every flush is atomic
    section,  // owners whose target only one chunk of their section reaches
    all,      // every owner: no two concurrently running chunks share a target
};

// Return the largest block multiple, up to the requested chunk size, whose
// worst-case staging fits the shared-memory limit, or 0 if none does.
int staged_chunk_size(const ResolvedInput& resolved,
                      const HierPlanOptions& options);

// Chunk each section and stage every grouped reference of each chunk: a slot
// per distinct target, owned by its first reference.
void build_staging(const ResolvedInput& resolved,
                   std::span<const ExecutionSection> sections,
                   int chunk_size, ExclusiveOwners exclusive, HierPlan& plan);

// Resolve the global target an argument reaches from one source element.
inline int arg_target(const op_arg& arg, int source) {
    return arg.map_data[static_cast<std::size_t>(source) *
                            static_cast<std::size_t>(arg.map->dim) +
                        static_cast<std::size_t>(arg.idx)];
}

} // namespace op::f2c::detail
