#include "op_hier_plan_internal.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace op::f2c {
namespace {

struct PlanOwner {
    void *owner;
    HierPlanReleaseCallback release;
};

std::mutex plan_owners_mutex;
std::vector<PlanOwner> plan_owners;

// Return the native size of a translated scalar type.
std::size_t scalar_size(HierScalarType type) {
    switch (type) {
    case HierScalarType::f32:
        return sizeof(float);
    case HierScalarType::f64:
        return sizeof(double);
    case HierScalarType::i32:
        return sizeof(int);
    }

    assert(false);
    return 0;
}

// Select and block-align the chunk size requested for this kernel.
int normalize_chunk_size(const HierArgGroups& groups,
                         const HierPlanOptions& options) {
    assert(options.block_size > 0);

    int requested = groups.chunk_size_override > 0
                        ? groups.chunk_size_override
                        : options.requested_chunk_size;
    if (requested <= 0)
        requested = options.block_size;

    // Round up to a block multiple, clamped so the result stays an int.
    long long blocks = (static_cast<long long>(requested) +
                        options.block_size - 1) / options.block_size;
    long long limit = std::numeric_limits<int>::max() / options.block_size;
    return static_cast<int>(std::max(1LL, std::min(blocks, limit)) *
                            options.block_size);
}

} // namespace

namespace detail {

HierFallbackReason resolve_input(
    op_set set, std::span<const op_arg> args,
    const HierArgGroups& groups,
    const HierPlanOptions& options, bool increments_only,
    ResolvedInput& resolved) {
    // The groups are generated alongside the wrapper they describe, so their
    // shape is an internal invariant rather than a runtime condition.
    assert(set != nullptr);
    assert(!groups.args.empty() && !groups.dats.empty());
    assert(groups.args.size() <= args.size());

    resolved.set_stride = (set->size + set->exec_size + 31) & ~31;
    resolved.requested_chunk_size = normalize_chunk_size(groups, options);
    resolved.args = args;
    resolved.arg_descriptors = groups.args;
    resolved.dats.resize(groups.dats.size());

    // Scalar layout is fixed by translation; runtime arguments supply the
    // active dat identity and dimension.
    for (std::size_t dat_index = 0; dat_index < groups.dats.size();
         ++dat_index)
        resolved.dats[dat_index].scalar_size =
            scalar_size(groups.dats[dat_index]);

    // Resolve optional state and the runtime identities behind each group.
    std::vector<bool> grouped_args(args.size(), false);
    bool any_active = false;
    for (std::size_t grouped_arg = 0; grouped_arg < groups.args.size();
         ++grouped_arg) {
        const auto& arg_desc = groups.args[grouped_arg];
        assert(arg_desc.arg_index >= 0);
        assert(arg_desc.dat_index >= 0);

        const auto arg_index = static_cast<std::size_t>(arg_desc.arg_index);
        const auto dat_index = static_cast<std::size_t>(arg_desc.dat_index);
        assert(arg_index < args.size() && dat_index < groups.dats.size());
        assert(!grouped_args[arg_index]);
        grouped_args[arg_index] = true;

        const op_arg& arg = args[arg_index];
        if (arg.opt == 0)
            continue;

        // op_arg_dat_core copies dim/size straight from the dat, and the
        // translator emitted these groups from the same parse as the
        // wrapper, so only the runtime identity is worth resolving here.
        assert(arg.argtype == OP_ARG_DAT);
        assert(!increments_only || arg.acc == OP_INC);
        assert(arg.dat != nullptr && arg.map != nullptr &&
               arg.map_data != nullptr);
        assert(arg.idx >= 0 && arg.idx < arg.map->dim);

        any_active = true;
        auto& resolved_dat = resolved.dats[dat_index];
        op_set target_set = arg.dat->set;

        if (resolved_dat.dat == nullptr) {
            resolved_dat.dat = arg.dat;
            resolved_dat.dimension = arg.dim;
            resolved_dat.target_extent = target_set->size +
                                         target_set->exec_size +
                                         target_set->nonexec_size;
        } else if (resolved_dat.dat != arg.dat) {
            return HierFallbackReason::incompatible_argument;
        }
    }

    if (!any_active)
        return HierFallbackReason::no_active_argument;

    // The translator groups dat arguments by source expression, so two groups
    // can still resolve to one runtime op_dat (the same dat passed through two
    // dummy arguments, say).  Each group is planned from its own targets: a
    // staged group owns an independent shared region and decides exclusivity
    // alone, and a coloured group's conflicts are found only within it, so
    // two groups on one dat could race on the same global address.
    //
    // An ungrouped argument on a grouped dat is a related hazard.  Colouring
    // does not order it, and under staging its reads would miss increments
    // still sitting in shared memory.  That loop is already unsound under the
    // baseline's global atomics, but staging turns "possibly stale" into
    // "certainly stale", so fall back rather than change its behaviour.
    //
    // One pass answers both: every dat argument must map to at most one group.
    std::unordered_map<const op_dat_core *, std::size_t> claimed;
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        const auto *dat = resolved.dats[dat_index].dat;
        if (dat != nullptr &&
            !claimed.emplace(dat, dat_index).second)
            return HierFallbackReason::incompatible_argument;
    }

    for (std::size_t arg_index = 0; arg_index < args.size(); ++arg_index) {
        const op_arg& arg = args[arg_index];
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT || arg.dat == nullptr)
            continue;

        if (!grouped_args[arg_index] && claimed.count(arg.dat) != 0)
            return HierFallbackReason::incompatible_argument;
    }

    return HierFallbackReason::none;
}

} // namespace detail

// Provide stable names for policy diagnostics and fallback reporting.
std::string_view
hier_fallback_reason_name(HierFallbackReason reason) {
    switch (reason) {
    case HierFallbackReason::none:
        return "none";
    case HierFallbackReason::disabled:
        return "disabled";
    case HierFallbackReason::unvalidated_device:
        return "unvalidated_device";
    case HierFallbackReason::not_staged:
        return "not_staged";
    case HierFallbackReason::no_active_argument:
        return "no_active_argument";
    case HierFallbackReason::incompatible_argument:
        return "incompatible_argument";
    case HierFallbackReason::insufficient_shared_memory:
        return "insufficient_shared_memory";
    case HierFallbackReason::low_compression:
        return "low_compression";
    case HierFallbackReason::too_many_colours:
        return "too_many_colours";
    }

    return "unknown";
}


void register_hier_plan_owner(void *owner,
                              HierPlanReleaseCallback release) {
    assert(owner != nullptr && release != nullptr);
    std::scoped_lock lock(plan_owners_mutex);
    assert(std::none_of(plan_owners.begin(), plan_owners.end(),
                        [owner](const PlanOwner& item) {
                            return item.owner == owner;
                        }));
    plan_owners.push_back({owner, release});
}

void unregister_hier_plan_owner(void *owner) {
    std::scoped_lock lock(plan_owners_mutex);
    std::erase_if(plan_owners, [owner](const PlanOwner& item) {
        return item.owner == owner;
    });
}

void release_hier_plan_device_storage() {
    std::vector<PlanOwner> owners;
    {
        std::scoped_lock lock(plan_owners_mutex);
        owners = plan_owners;
    }

    for (const auto& owner : owners)
        owner.release(owner.owner);
}

} // namespace op::f2c
