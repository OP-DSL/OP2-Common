#include <plan/staging.h>

#include <plan/greedy_colour.h>

#include <algorithm>
#include <cassert>
#include <limits>

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

} // namespace

std::size_t scratch_bytes(const References& references, std::span<const int> counts) {
    std::size_t bytes = 0;
    for (std::size_t group = 0; group < references.groups.size(); ++group) {
        // Inactive groups keep their place with a zero-length region, so the
        // staged kernel can walk this layout without branching on opt state.
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

int staged_chunk_size(const References& references, int size, int lanes,
                      std::size_t scratch_bytes) {
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
    std::size_t limit = scratch_bytes > slack ? scratch_bytes - slack : 0;

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

Staging stage(const Chunks& chunks, const ChunkTargets& targets,
              const References& references, int stride, bool unstaged_singles) {
    Staging staging;
    staging.stride = stride;
    const std::size_t groups = references.groups.size();
    staging.stage_words.assign(references.items.size() * static_cast<std::size_t>(stride), 0);
    staging.stage_counts.reserve(static_cast<std::size_t>(chunks.count()) * groups);
    staging.slot_offsets.reserve(static_cast<std::size_t>(chunks.count()) * groups);
    staging.chunk_scratch_bytes.reserve(static_cast<std::size_t>(chunks.count()));

    // Per group, each of the chunk's targets' slot.
    std::vector<std::vector<int>> slots(groups);
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
            staging.stage_counts.push_back(count);
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

        staging.chunk_scratch_bytes.push_back(scratch_bytes(
            references, std::span<const int>{staging.stage_counts}.last(groups)));
    }
    return staging;
}

} // namespace op::plan
