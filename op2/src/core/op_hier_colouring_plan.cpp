#include "op_hier_plan_internal.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <utility>

namespace op::f2c {
namespace {

using detail::arg_target;
using detail::ResolvedInput;

// One target of a coloured argument: a dat group and an element of its set.
struct ColourTarget {
    std::size_t dat;
    int target;
};

// Colour items greedily so that no two items of a colour share a target,
// trying 64 colours a round.  targets(i) lists item i's targets, and masks,
// one per target of each dat, start and end clear.  Return the colour count.
template<typename Targets>
int greedy_colour(int count, Targets&& targets,
                  std::vector<std::vector<std::uint64_t>>& masks,
                  std::vector<int>& colours) {
    colours.assign(static_cast<std::size_t>(count), -1);
    std::vector<ColourTarget> marked;
    int coloured = 0;
    int ncolours = 0;

    for (int base = 0; coloured < count; base += 64) {
        for (int item = 0; item < count; ++item) {
            auto& colour = colours[static_cast<std::size_t>(item)];
            if (colour >= 0)
                continue;

            std::uint64_t used = 0;
            for (const auto& [dat, target] : targets(item))
                used |= masks[dat][static_cast<std::size_t>(target)];
            if (used == ~std::uint64_t{0})
                continue;

            int bit = std::countr_one(used);
            colour = base + bit;
            ncolours = std::max(ncolours, colour + 1);
            ++coloured;

            for (const auto& [dat, target] : targets(item)) {
                auto& mask = masks[dat][static_cast<std::size_t>(target)];
                if (mask == 0)
                    marked.push_back({dat, target});
                mask |= std::uint64_t{1} << bit;
            }
        }

        for (const auto& [dat, target] : marked)
            masks[dat][static_cast<std::size_t>(target)] = 0;
        marked.clear();
    }

    return ncolours;
}


} // namespace

// Colour each section's chunks, and each chunk's elements, over the targets of
// the coloured arguments.
HierPlanBuildResult build_hier_colouring_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierArgGroups& groups,
    const HierPlanOptions& options) {
    ResolvedInput resolved;
    auto reason = resolve_input(set, args, groups, options, false, resolved);
    if (reason != HierFallbackReason::none)
        return {reason, std::nullopt};

    assert(!sections.empty() && sections.front().start == 0);
    assert(sections.back().end == set->size + set->exec_size);

    // The active coloured arguments, each with its dat group.
    std::vector<std::pair<const op_arg *, std::size_t>> coloured;
    for (const auto& arg_desc : groups.args) {
        const op_arg& arg = args[static_cast<std::size_t>(arg_desc.arg_index)];
        if (arg.opt != 0)
            coloured.push_back(
                {&arg, static_cast<std::size_t>(arg_desc.dat_index)});
    }
    const std::size_t per_element = coloured.size();

    std::vector<std::vector<std::uint64_t>> masks(resolved.dats.size());
    std::vector<std::vector<std::uint8_t>> seen(resolved.dats.size());
    for (std::size_t dat = 0; dat < resolved.dats.size(); ++dat) {
        if (resolved.dats[dat].dat == nullptr)
            continue;
        auto extent =
            static_cast<std::size_t>(resolved.dats[dat].target_extent);
        masks[dat].assign(extent, 0);
        seen[dat].assign(extent, 0);
    }

    const int chunk_size = resolved.requested_chunk_size;
    HierPlan plan;
    plan.selected_chunk_size = chunk_size;
    plan.set_stride = resolved.set_stride;
    auto& colouring = plan.colouring.emplace();
    colouring.thread_colours.assign(static_cast<std::size_t>(plan.set_stride), 0);
    plan.source_offsets.push_back(0);
    plan.section_chunk_offsets.push_back(0);
    colouring.section_launch_offsets.push_back(0);
    colouring.launch_chunk_offsets.push_back(0);

    std::vector<ColourTarget> element_targets;
    std::vector<ColourTarget> chunk_targets;
    std::vector<std::size_t> chunk_target_offsets;
    std::vector<int> colours;

    for (const ExecutionSection& section : sections) {
        const int first_chunk = static_cast<int>(plan.num_chunks());
        chunk_targets.clear();
        chunk_target_offsets.assign(1, 0);

        for (int start = section.start; start < section.end;) {
            int end = static_cast<int>(std::min(
                static_cast<long long>(section.end),
                static_cast<long long>(start) + chunk_size));

            // Every element's targets, and the chunk's distinct ones.
            element_targets.clear();
            for (int source = start; source < end; ++source) {
                for (const auto& [arg, dat] : coloured) {
                    int target = arg_target(*arg, source);
                    assert(target >= 0 &&
                           target < resolved.dats[dat].target_extent);
                    element_targets.push_back({dat, target});

                    auto& flag = seen[dat][static_cast<std::size_t>(target)];
                    if (flag == 0) {
                        flag = 1;
                        chunk_targets.push_back({dat, target});
                    }
                }
            }
            for (std::size_t i = chunk_target_offsets.back();
                 i < chunk_targets.size(); ++i)
                seen[chunk_targets[i].dat][
                    static_cast<std::size_t>(chunk_targets[i].target)] = 0;

            plan.statistics.raw_references += element_targets.size();
            plan.statistics.distinct_targets +=
                chunk_targets.size() - chunk_target_offsets.back();
            chunk_target_offsets.push_back(chunk_targets.size());

            int thread_colours = greedy_colour(
                end - start,
                [&](int element) {
                    return std::span<const ColourTarget>{
                        element_targets.data() +
                            static_cast<std::size_t>(element) * per_element,
                        per_element};
                },
                masks, colours);
            if (thread_colours > 255)
                return {HierFallbackReason::too_many_colours,
                        std::nullopt};

            for (int source = start; source < end; ++source)
                colouring.thread_colours[static_cast<std::size_t>(source)] =
                    static_cast<std::uint8_t>(
                        colours[static_cast<std::size_t>(source - start)]);
            colouring.chunk_thread_colours.push_back(thread_colours);
            plan.statistics.max_thread_colours =
                std::max(plan.statistics.max_thread_colours, thread_colours);

            plan.source_offsets.push_back(end);
            start = end;
        }

        // Colour the section's chunks, then list them launch by launch.
        const int nchunks = static_cast<int>(plan.num_chunks()) - first_chunk;
        int block_colours = greedy_colour(
            nchunks,
            [&](int chunk) {
                auto begin = chunk_target_offsets[static_cast<std::size_t>(chunk)];
                auto end = chunk_target_offsets[static_cast<std::size_t>(chunk) + 1];
                return std::span<const ColourTarget>{
                    chunk_targets.data() + begin, end - begin};
            },
            masks, colours);

        std::vector<int> launch_sizes(static_cast<std::size_t>(block_colours), 0);
        for (int chunk = 0; chunk < nchunks; ++chunk)
            ++launch_sizes[static_cast<std::size_t>(
                colours[static_cast<std::size_t>(chunk)])];

        std::vector<int> launch_next(static_cast<std::size_t>(block_colours));
        int next = static_cast<int>(colouring.chunk_order.size());
        for (int colour = 0; colour < block_colours; ++colour) {
            launch_next[static_cast<std::size_t>(colour)] = next;
            next += launch_sizes[static_cast<std::size_t>(colour)];
            colouring.launch_chunk_offsets.push_back(next);
        }

        colouring.chunk_order.resize(static_cast<std::size_t>(next));
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            auto colour = colours[static_cast<std::size_t>(chunk)];
            colouring.chunk_order[static_cast<std::size_t>(
                launch_next[static_cast<std::size_t>(colour)]++)] =
                first_chunk + chunk;
        }

        plan.section_chunk_offsets.push_back(
            static_cast<int>(plan.num_chunks()));
        colouring.section_launch_offsets.push_back(
            static_cast<int>(colouring.launch_chunk_offsets.size()) - 1);
    }

    plan.statistics.launches = colouring.launch_chunk_offsets.size() - 1;
    return {HierFallbackReason::none, std::move(plan)};
}

} // namespace op::f2c
