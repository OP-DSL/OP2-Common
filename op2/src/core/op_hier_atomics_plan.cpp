#include "op_hier_plan_internal.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace op::f2c {
namespace {

using detail::arg_target;
using detail::ResolvedInput;

// Lay out one chunk's heterogeneous dat regions and return their total bytes.
// Every term is bounded by the shared-memory capacity the chunk was sized
// against, so plain size_t arithmetic cannot wrap here.
std::size_t calculate_shared_bytes(
    const ResolvedInput& resolved,
    const std::vector<std::vector<int>>& touched) {
    std::size_t shared_bytes = 0;
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        // Inactive groups keep their place with a zero-length region, so the
        // staged wrapper can walk this layout without branching on opt state.
        const auto& dat = resolved.dats[dat_index];
        std::size_t alignment = dat.scalar_size;
        shared_bytes = (shared_bytes + alignment - 1) & ~(alignment - 1);
        shared_bytes += touched[dat_index].size() *
                        static_cast<std::size_t>(dat.dimension) *
                        dat.scalar_size;
    }

    return shared_bytes;
}

// Return the padding the region layout can insert.  The first region starts at
// offset zero, so only the regions after it can need aligning.
std::size_t alignment_slack(const ResolvedInput& resolved) {
    std::size_t slack = 0;
    for (std::size_t dat_index = 1; dat_index < resolved.dats.size();
         ++dat_index)
        slack += resolved.dats[dat_index].scalar_size - 1;

    return slack;
}

// Return the most shared bytes a single source element can require, which is
// what bounds the chunk size before any map is inspected.
std::size_t bytes_per_source_element(const ResolvedInput& resolved) {
    std::vector<std::size_t> args_per_dat(resolved.dats.size(), 0);
    for (const auto& arg_desc : resolved.arg_descriptors)
        if (resolved.args[static_cast<std::size_t>(arg_desc.arg_index)].opt != 0)
            ++args_per_dat[static_cast<std::size_t>(arg_desc.dat_index)];

    std::size_t total = 0;
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        const auto& dat = resolved.dats[dat_index];
        if (dat.dat == nullptr)
            continue;

        // Each active argument can reach one distinct target per element.
        total += args_per_dat[dat_index] *
                 static_cast<std::size_t>(dat.dimension) * dat.scalar_size;
    }

    return total;
}

// Construct the complete host plan for one fixed candidate chunk size.
void build_candidate(const ResolvedInput& resolved,
                     std::span<const ExecutionSection> sections,
                     int chunk_size, bool exclusive_flush,
                     HierPlan& plan) {
    plan = {};
    plan.selected_chunk_size = chunk_size;
    plan.set_stride = resolved.set_stride;
    auto& staging = plan.staging.emplace();

    // Pre-size the persistent arrays for this candidate.
    std::size_t num_chunks = 0;
    for (const auto& section : sections)
        num_chunks += static_cast<std::size_t>(
            (static_cast<long long>(section.size()) + chunk_size - 1) /
            chunk_size);

    staging.stage_words.assign(
        resolved.arg_descriptors.size() *
            static_cast<std::size_t>(plan.set_stride),
        0);
    staging.stage_counts.reserve(num_chunks * resolved.dats.size());
    plan.source_offsets.reserve(num_chunks + 1);
    plan.section_chunk_offsets.reserve(sections.size() + 1);
    staging.section_shared_bytes.reserve(sections.size());

    // Allocate one reusable target-to-slot inverse map per active dat, plus
    // the per-section chunk tally that decides exclusivity.
    std::vector<std::vector<int>> inverse(resolved.dats.size());
    std::vector<std::vector<int>> touched(resolved.dats.size());
    std::vector<std::vector<int>> section_chunks(resolved.dats.size());
    std::vector<std::vector<int>> section_targets(resolved.dats.size());
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        if (resolved.dats[dat_index].dat != nullptr) {
            const auto target_extent = static_cast<std::size_t>(
                resolved.dats[dat_index].target_extent);
            inverse[dat_index].assign(target_extent, -1);
            section_chunks[dat_index].assign(target_extent, 0);
        }
    }

    plan.source_offsets.push_back(0);
    plan.section_chunk_offsets.push_back(0);

    for (const ExecutionSection& section : sections) {
        std::size_t section_shared_bytes = 0;

        // Chunk each schedule section independently so launches retain their
        // halo-wait and global-processing boundaries.
        for (int start = section.start; start < section.end;) {
            int end = static_cast<int>(std::min(
                static_cast<long long>(section.end),
                static_cast<long long>(start) + chunk_size));

            // Reset only inverse-map entries touched by the previous chunk.
            for (std::size_t dat_index = 0; dat_index < touched.size();
                 ++dat_index) {
                for (int target : touched[dat_index])
                    inverse[dat_index][static_cast<std::size_t>(target)] = -1;
                touched[dat_index].clear();
            }

            // Assign shared slots and mark the first reference as its owner.
            for (int source = start; source < end; ++source) {
                for (std::size_t staged_arg = 0;
                     staged_arg < resolved.arg_descriptors.size();
                     ++staged_arg) {
                    const auto& arg_desc =
                        resolved.arg_descriptors[staged_arg];
                    const op_arg& arg = resolved.args[static_cast<std::size_t>(
                        arg_desc.arg_index)];
                    if (arg.opt == 0)
                        continue;

                    const auto dat_index = static_cast<std::size_t>(
                        arg_desc.dat_index);
                    int target = arg_target(arg, source);
                    // op_decl_map validates targets, and the baseline wrapper
                    // indexes just as blindly as this does.
                    assert(target >= 0 &&
                           target < resolved.dats[dat_index].target_extent);

                    int& slot =
                        inverse[dat_index][static_cast<std::size_t>(target)];
                    bool owner = slot < 0;
                    if (owner) {
                        slot = static_cast<int>(touched[dat_index].size());
                        touched[dat_index].push_back(target);

                        // One tally per chunk that reaches this target: a
                        // target seen in a single chunk of the section can be
                        // flushed without a global atomic.
                        auto& chunks = section_chunks[dat_index][
                            static_cast<std::size_t>(target)];
                        if (chunks == 0)
                            section_targets[dat_index].push_back(target);
                        ++chunks;
                    }

                    std::size_t word_index =
                        staged_arg * static_cast<std::size_t>(plan.set_stride) +
                        static_cast<std::size_t>(source);
                    staging.stage_words[word_index] = hier_smem_pack_stage_word(
                        static_cast<std::uint32_t>(slot), owner);

                    ++plan.statistics.raw_references;
                }
            }

            // Record exact region sizes and statistics for this chunk.
            for (std::size_t dat_index = 0; dat_index < touched.size();
                 ++dat_index) {
                staging.stage_counts.push_back(
                    static_cast<int>(touched[dat_index].size()));

                plan.statistics.distinct_targets += touched[dat_index].size();
            }

            section_shared_bytes = std::max(
                section_shared_bytes, calculate_shared_bytes(resolved, touched));

            plan.source_offsets.push_back(end);
            start = end;
        }

        // Every chunk in this section is now counted, so a second pass can
        // mark the owners whose target this section reaches exactly once.
        // Chunks within a section may run concurrently on different blocks;
        // separate sections are ordered on the stream, so they cannot race.
        if (exclusive_flush) {
            for (int source = section.start; source < section.end; ++source) {
                for (std::size_t staged_arg = 0;
                     staged_arg < resolved.arg_descriptors.size();
                     ++staged_arg) {
                    const auto& arg_desc =
                        resolved.arg_descriptors[staged_arg];
                    const op_arg& arg = resolved.args[static_cast<std::size_t>(
                        arg_desc.arg_index)];
                    if (arg.opt == 0)
                        continue;

                    std::size_t word_index =
                        staged_arg * static_cast<std::size_t>(plan.set_stride) +
                        static_cast<std::size_t>(source);
                    auto word = staging.stage_words[word_index];
                    if (!hier_smem_stage_owner(word))
                        continue;

                    const auto dat_index = static_cast<std::size_t>(
                        arg_desc.dat_index);
                    int target = arg_target(arg, source);
                    if (section_chunks[dat_index][
                            static_cast<std::size_t>(target)] != 1)
                        continue;

                    staging.stage_words[word_index] = word |
                                                   hier_smem_exclusive_bit;
                    ++plan.statistics.exclusive_owners;
                }
            }
        }

        // Clear only the targets this section reached.
        for (std::size_t dat_index = 0; dat_index < section_targets.size();
             ++dat_index) {
            for (int target : section_targets[dat_index])
                section_chunks[dat_index][
                    static_cast<std::size_t>(target)] = 0;
            section_targets[dat_index].clear();
        }

        plan.section_chunk_offsets.push_back(
            static_cast<int>(plan.num_chunks()));
        staging.section_shared_bytes.push_back(section_shared_bytes);
    }
}

} // namespace

// Build the largest block-aligned plan that fits the shared-memory limit.
HierPlanBuildResult build_hier_atomics_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierArgGroups& groups,
    const HierPlanOptions& options) {
    // Resolve runtime metadata once for every candidate.
    ResolvedInput resolved;
    auto reason = resolve_input(set, args, groups, options, true, resolved);
    if (reason != FallbackReason::none)
        return {reason, std::nullopt};

    assert(!sections.empty() && sections.front().start == 0);
    assert(sections.back().end == set->size + set->exec_size);

    // A chunk's distinct targets cannot outnumber its source references, so
    // the exact plan is bounded by chunk_size * bytes_per_source_element.
    // Solving that for chunk_size picks a size guaranteed to fit up front,
    // and the plan is then built exactly once.  Region alignment adds at most
    // one scalar of padding per staged dat, which comes off the limit first.
    std::size_t slack = alignment_slack(resolved);
    std::size_t limit = options.shared_memory_limit > slack
                            ? options.shared_memory_limit - slack
                            : 0;

    std::size_t per_element = bytes_per_source_element(resolved);
    assert(per_element > 0);

    auto capacity = static_cast<long long>(limit / per_element);
    long long blocks = capacity / options.block_size;
    if (blocks < 1)
        return {FallbackReason::insufficient_shared_memory,
                std::nullopt};

    int chunk_size = static_cast<int>(std::min(
        static_cast<long long>(resolved.requested_chunk_size),
        blocks * options.block_size));

    HierPlan plan;
    build_candidate(resolved, sections, chunk_size, options.exclusive_flush,
                    plan);
    plan.staging->has_exclusive = plan.statistics.exclusive_owners > 0 ? 1 : 0;
    return {FallbackReason::none, std::move(plan)};
}


} // namespace op::f2c
