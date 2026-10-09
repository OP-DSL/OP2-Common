#include "op_hier_plan_internal.h"

#include <cassert>
#include <utility>

namespace op::f2c {

// Build the largest block-aligned staging plan that fits the shared-memory
// limit.  Concurrent chunks of a section may share targets, so only owners
// whose target one chunk of the section reaches can flush without atomics.
HierPlanBuildResult build_hier_atomics_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierArgGroups& groups,
    const HierPlanOptions& options) {
    detail::ResolvedInput resolved;
    auto reason = detail::resolve_input(set, args, groups, options, true,
                                        resolved);
    if (reason != FallbackReason::none)
        return {reason, std::nullopt};

    assert(!sections.empty() && sections.front().start == 0);
    assert(sections.back().end == set->size + set->exec_size);

    int chunk_size = detail::staged_chunk_size(resolved, options);
    if (chunk_size == 0)
        return {FallbackReason::insufficient_shared_memory, std::nullopt};

    HierPlan plan;
    detail::build_staging(resolved, sections, chunk_size,
                          options.exclusive_flush
                              ? detail::ExclusiveOwners::section
                              : detail::ExclusiveOwners::none,
                          plan);
    plan.staging->has_exclusive = plan.statistics.exclusive_owners > 0 ? 1 : 0;
    return {FallbackReason::none, std::move(plan)};
}

} // namespace op::f2c
