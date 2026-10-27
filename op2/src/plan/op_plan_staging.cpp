#include <plan/staging.h>

#include <plan/greedy_colour.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <unordered_map>

namespace op::plan {

using detail::arg_target;

namespace {

// The padding the region layout can insert.  The first region starts at
// offset zero, so only the regions after it can need aligning.
std::size_t alignment_slack(const References& references) {
    std::size_t slack = 0;
    for (std::size_t group = 1; group < references.groups.size(); ++group)
        slack += references.groups[group].scalar_size > 0
                     ? references.groups[group].scalar_size - 1
                     : 0;
    return slack;
}

// The active references to each group.
std::vector<int> references_per_group(const References& references) {
    std::vector<int> counts(references.groups.size(), 0);
    for (const Reference& reference : references.items)
        if (reference.arg->opt != 0)
            ++counts[static_cast<std::size_t>(reference.group)];
    return counts;
}

// The most scratch bytes a single source element can require, which is what
// bounds the chunk size before any map is inspected.
std::size_t bytes_per_source_element(const References& references) {
    auto counts = references_per_group(references);
    std::size_t total = 0;
    for (std::size_t group = 0; group < references.groups.size(); ++group) {
        const auto& resolved = references.groups[group];
        if (resolved.dat == nullptr)
            continue;
        // Each active reference can reach one distinct target per element.
        total += static_cast<std::size_t>(counts[group]) *
                 static_cast<std::size_t>(resolved.dimension) * resolved.scalar_size;
    }
    return total;
}

// The scratch bytes of a chunk with counts slots per group: the regions in
// group order, each aligned to its scalar size.
std::size_t region_bytes(const References& references, std::span<const int> counts) {
    std::size_t bytes = 0;
    for (std::size_t group = 0; group < references.groups.size(); ++group) {
        // An inactive group takes no space; the kernel walks it as an empty,
        // unpadded region.
        const auto& resolved = references.groups[group];
        if (resolved.dat == nullptr)
            continue;
        std::size_t alignment = resolved.scalar_size;
        bytes = (bytes + alignment - 1) & ~(alignment - 1);
        bytes += static_cast<std::size_t>(counts[group]) *
                 static_cast<std::size_t>(resolved.dimension) * resolved.scalar_size;
    }
    return bytes;
}

// The largest multiple of lanes, up to size rounded up to lanes, whose
// chunks' worst-case staging fits scratch_limit and the stage word; 0 if
// none does.
int staged_chunk_size(const References& references, int size, int lanes,
                      std::size_t scratch_limit) {
    assert(size > 0 && lanes > 0);

    // Round the requested size up to a lanes multiple, clamped so the result
    // stays an int.
    long long requested = (static_cast<long long>(size) + lanes - 1) / lanes;
    requested = std::max(1LL, std::min(requested,
                                       static_cast<long long>(
                                           std::numeric_limits<int>::max() / lanes)));
    requested *= lanes;

    // A chunk's distinct targets cannot outnumber its source references, so
    // the exact staging is bounded by size * bytes_per_source_element.
    // Solving that for size picks one guaranteed to fit up front, so the
    // staging is built exactly once.  Region alignment adds at most one
    // scalar of padding per staged dat, which comes off the limit first.
    std::size_t slack = alignment_slack(references);
    std::size_t limit = scratch_limit > slack ? scratch_limit - slack : 0;

    std::size_t per_element = bytes_per_source_element(references);
    assert(per_element > 0);

    // A region's slots must also fit the stage word.
    auto counts = references_per_group(references);
    int per_group = std::max(*std::max_element(counts.begin(), counts.end()), 1);
    auto capacity = std::min(static_cast<long long>(limit / per_element),
                             static_cast<long long>(max_slots / per_group));
    long long blocks = capacity / lanes;
    if (blocks < 1)
        return 0;
    return static_cast<int>(std::min(requested, blocks * lanes));
}

// Stage every active reference of each chunk.
Staging stage(const Chunks& chunks, const ChunkTargets& targets,
              const References& references, int stride, bool unstaged_singles) {
    Staging staging;
    staging.stride = stride;
    const std::size_t groups = references.groups.size();
    staging.stage_words.assign(references.items.size() * static_cast<std::size_t>(stride), 0);
    staging.slot_offsets.reserve(static_cast<std::size_t>(chunks.count()) * groups + 1);
    staging.chunk_scratch_bytes.reserve(static_cast<std::size_t>(chunks.count()));

    // Per group, each of the chunk's targets' slot, and the chunk's slot count.
    std::vector<std::vector<int>> slots(groups);
    std::vector<int> counts(groups);
    for (std::size_t group = 0; group < groups; ++group)
        slots[group].assign(static_cast<std::size_t>(references.groups[group].extent), -1);

    for (int chunk = 0; chunk < chunks.count(); ++chunk) {
        // Slots in target order, as op_plan's ind_map has them, so a lane per
        // slot seeds and flushes ascending addresses.  With unstaged_singles
        // a target referenced once stays in global memory: staging it would
        // save no atomic.
        for (std::size_t group = 0; group < groups; ++group) {
            auto region = targets.region(chunk, static_cast<int>(group));
            staging.slot_offsets.push_back(static_cast<int>(staging.slot_targets.size()));
            int count = 0;
            for (int entry = targets.offsets[region]; entry < targets.offsets[region + 1];
                 ++entry) {
                int target = targets.targets[static_cast<std::size_t>(entry)];
                int& slot = slots[group][static_cast<std::size_t>(target)];
                if (unstaged_singles && targets.references[static_cast<std::size_t>(entry)] == 1) {
                    slot = unstaged_word;
                    ++staging.unstaged_singles;
                    continue;
                }
                slot = count++;
                staging.slot_targets.push_back(target);
            }
            counts[group] = count;
        }

        for (int element = chunks.begin(chunk); element < chunks.end(chunk); ++element)
            for (std::size_t index = 0; index < references.items.size(); ++index) {
                const Reference& reference = references.items[index];
                if (reference.arg->opt == 0)
                    continue;
                int target = arg_target(*reference.arg, element);
                staging.stage_words[index * static_cast<std::size_t>(stride) +
                                    static_cast<std::size_t>(element)] =
                    static_cast<StageWord>(
                        slots[static_cast<std::size_t>(reference.group)]
                             [static_cast<std::size_t>(target)]);
            }

        staging.chunk_scratch_bytes.push_back(region_bytes(references, counts));
    }
    staging.slot_offsets.push_back(static_cast<int>(staging.slot_targets.size()));
    return staging;
}

bool indirect_update(const op_arg& arg, bool read_write) {
    return arg.idx >= 0 && arg.map != nullptr &&
           (arg.acc == OP_INC || (read_write && arg.acc == OP_RW));
}

// The groups' arguments as references, or why the call does not suit them
// (stage_chunks' reasons but insufficient_scratch).
std::string_view grouped_references(std::span<const op_arg> args,
                                    const ArgGroups& groups, bool read_write,
                                    References& references) {
    // The groups are generated alongside the kernel they describe, so their
    // shape is an internal invariant rather than a runtime condition.
    assert(!groups.args.empty() && groups.groups > 0);
    assert(groups.args.size() <= args.size());

    references = {};
    references.groups.assign(static_cast<std::size_t>(groups.groups), {});
    std::vector<bool> grouped(args.size(), false);
    bool any_active = false;
    for (const ArgGroup& group : groups.args) {
        assert(group.arg >= 0 && group.group >= 0);
        const auto arg_index = static_cast<std::size_t>(group.arg);
        const auto group_index = static_cast<std::size_t>(group.group);
        assert(arg_index < args.size() && group_index < references.groups.size());
        assert(!grouped[arg_index]);
        grouped[arg_index] = true;

        const op_arg& arg = args[arg_index];
        references.items.push_back({&arg, group.group});
        if (arg.opt == 0)
            continue;

        assert(arg.argtype == OP_ARG_DAT);
        assert(read_write || arg.acc == OP_INC);
        assert(arg.dat != nullptr && arg.map != nullptr);
        // A set with no elements to run may have no map data.
        assert(arg.map_data != nullptr || arg.map->from->size + arg.map->from->exec_size == 0);
        assert(arg.idx >= 0 && arg.idx < arg.map->dim);
        assert(arg.dim > 0 && arg.size % arg.dim == 0);

        any_active = true;
        auto& resolved = references.groups[group_index];
        if (resolved.dat == nullptr)
            resolved = Group::of(arg);
        else if (resolved.dat != arg.dat)
            return "incompatible_argument";
    }

    if (!any_active)
        return "no_active_argument";

    // Groups are planned apart, so two resolving to one dat (passed through two
    // dummy arguments) would race.
    std::unordered_map<const op_dat_core *, std::size_t> claimed;
    for (std::size_t index = 0; index < references.groups.size(); ++index) {
        const auto *dat = references.groups[index].dat;
        if (dat != nullptr && !claimed.emplace(dat, index).second)
            return "incompatible_argument";
    }

    // Nothing orders an ungrouped argument: on a grouped dat it would read
    // stale staged targets, and an indirect update would race.
    for (std::size_t index = 0; index < args.size(); ++index) {
        const op_arg& arg = args[index];
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT || arg.dat == nullptr ||
            grouped[index])
            continue;
        if (claimed.count(arg.dat) != 0 || indirect_update(arg, read_write))
            return "incompatible_argument";
    }

    return {};
}

} // namespace

std::string_view stage_chunks(std::span<const op_arg> args,
                              const op::loop::Sections& sections,
                              const ArgGroups& groups, bool read_write, int size,
                              int lanes, std::size_t scratch_limit,
                              bool unstaged_singles, StagedChunks& out) {
    auto reason = grouped_references(args, groups, read_write, out.references);
    if (!reason.empty())
        return reason;
    int chunk_size = staged_chunk_size(out.references, size, lanes, scratch_limit);
    if (chunk_size == 0)
        return "insufficient_scratch";

    out.chunks = chunk_sections(sections, chunk_size);
    out.targets = chunk_targets(out.chunks, out.references);
    out.staging = stage(out.chunks, out.targets, out.references, sections.set_stride(),
                        unstaged_singles);
    return {};
}

} // namespace op::plan
