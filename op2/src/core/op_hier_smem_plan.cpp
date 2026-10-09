#include <op_hier_smem_plan.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace op::f2c {
namespace {

struct PlanOwner {
    void *owner;
    HierSmemPlanReleaseCallback release;
};

std::mutex plan_owners_mutex;
std::vector<PlanOwner> plan_owners;

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
    std::span<const HierSmemArgDescriptor> arg_descriptors;
    std::vector<ResolvedDat> dats;
};

// Return the native size of a translated scalar type.
std::size_t scalar_size(HierSmemScalarType type) {
    switch (type) {
    case HierSmemScalarType::f32:
        return sizeof(float);
    case HierSmemScalarType::f64:
        return sizeof(double);
    case HierSmemScalarType::i32:
        return sizeof(int);
    }

    assert(false);
    return 0;
}

// Select and block-align the chunk size requested for this kernel.
int normalize_chunk_size(const HierSmemStagingDescriptor& descriptor,
                         const HierSmemPlanOptions& options) {
    assert(options.block_size > 0);

    int requested = descriptor.chunk_size_override > 0
                        ? descriptor.chunk_size_override
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

// Resolve generated metadata and runtime identities into canonical plan input.
HierSmemFallbackReason resolve_input(
    op_set set, std::span<const op_arg> args,
    const HierSmemStagingDescriptor& descriptor,
    const HierSmemPlanOptions& options, bool increments_only,
    ResolvedInput& resolved) {
    // The descriptor is generated alongside the wrapper it describes, so its
    // shape is an internal invariant rather than a runtime condition.
    assert(set != nullptr);
    assert(!descriptor.args.empty() && !descriptor.dats.empty());
    assert(descriptor.args.size() <= args.size());

    resolved.set_stride = (set->size + set->exec_size + 31) & ~31;
    resolved.requested_chunk_size = normalize_chunk_size(descriptor, options);
    resolved.args = args;
    resolved.arg_descriptors = descriptor.args;
    resolved.dats.resize(descriptor.dats.size());

    // Scalar layout is fixed by translation; runtime arguments supply the
    // active dat identity and dimension.
    for (std::size_t dat_index = 0; dat_index < descriptor.dats.size();
         ++dat_index)
        resolved.dats[dat_index].scalar_size =
            scalar_size(descriptor.dats[dat_index]);

    // Resolve optional state and the runtime identities behind each group.
    std::vector<bool> staged_args(args.size(), false);
    bool any_active = false;
    for (std::size_t staged_arg = 0; staged_arg < descriptor.args.size();
         ++staged_arg) {
        const auto& arg_desc = descriptor.args[staged_arg];
        assert(arg_desc.arg_index >= 0);
        assert(arg_desc.dat_index >= 0);

        const auto arg_index = static_cast<std::size_t>(arg_desc.arg_index);
        const auto dat_index = static_cast<std::size_t>(arg_desc.dat_index);
        assert(arg_index < args.size() && dat_index < descriptor.dats.size());
        assert(!staged_args[arg_index]);
        staged_args[arg_index] = true;

        const op_arg& arg = args[arg_index];
        if (arg.opt == 0)
            continue;

        // op_arg_dat_core copies dim/size straight from the dat, and the
        // translator emitted this descriptor from the same parse as the
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
            return HierSmemFallbackReason::incompatible_argument;
        }
    }

    if (!any_active)
        return HierSmemFallbackReason::no_active_increment;

    // The translator groups dat arguments by source expression, so two groups
    // can still resolve to one runtime op_dat (the same dat passed through two
    // dummy arguments, say).  Each group owns an independent shared region and
    // decides exclusivity from its own targets, so once exclusive owners flush
    // with a non-atomic +=, two groups can race on the same global address.
    //
    // A non-staged argument on a staged dat is a related hazard: its reads
    // would miss increments still sitting in shared memory.  That loop is
    // already unsound under the baseline's global atomics, but staging turns
    // "possibly stale" into "certainly stale", so fall back rather than change
    // its behaviour.
    //
    // One pass answers both: every dat argument must map to at most one group.
    std::unordered_map<const op_dat_core *, std::size_t> claimed;
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        const auto *dat = resolved.dats[dat_index].dat;
        if (dat != nullptr &&
            !claimed.emplace(dat, dat_index).second)
            return HierSmemFallbackReason::incompatible_argument;
    }

    for (std::size_t arg_index = 0; arg_index < args.size(); ++arg_index) {
        const op_arg& arg = args[arg_index];
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT || arg.dat == nullptr)
            continue;

        if (!staged_args[arg_index] && claimed.count(arg.dat) != 0)
            return HierSmemFallbackReason::incompatible_argument;
    }

    return HierSmemFallbackReason::none;
}

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

// Resolve the global target an argument reaches from one source element.
inline int staged_target(const op_arg& arg, int source) {
    return arg.map_data[static_cast<std::size_t>(source) *
                            static_cast<std::size_t>(arg.map->dim) +
                        static_cast<std::size_t>(arg.idx)];
}

// Construct the complete host plan for one fixed candidate chunk size.
void build_candidate(const ResolvedInput& resolved,
                     std::span<const ExecutionSection> sections,
                     int chunk_size, bool exclusive_flush,
                     HierSmemPlan& plan) {
    plan = {};
    plan.selected_chunk_size = chunk_size;
    plan.set_stride = resolved.set_stride;

    // Pre-size the persistent arrays for this candidate.
    std::size_t num_chunks = 0;
    for (const auto& section : sections)
        num_chunks += static_cast<std::size_t>(
            (static_cast<long long>(section.size()) + chunk_size - 1) /
            chunk_size);

    plan.stage_words.assign(
        resolved.arg_descriptors.size() *
            static_cast<std::size_t>(plan.set_stride),
        0);
    plan.stage_counts.reserve(num_chunks * resolved.dats.size());
    plan.source_offsets.reserve(num_chunks + 1);
    plan.section_chunk_offsets.reserve(sections.size() + 1);
    plan.section_shared_bytes.reserve(sections.size());

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
                    int target = staged_target(arg, source);
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
                    plan.stage_words[word_index] = hier_smem_pack_stage_word(
                        static_cast<std::uint32_t>(slot), owner);

                    ++plan.statistics.raw_references;
                }
            }

            // Record exact region sizes and statistics for this chunk.
            for (std::size_t dat_index = 0; dat_index < touched.size();
                 ++dat_index) {
                plan.stage_counts.push_back(
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
                    auto word = plan.stage_words[word_index];
                    if (!hier_smem_stage_owner(word))
                        continue;

                    const auto dat_index = static_cast<std::size_t>(
                        arg_desc.dat_index);
                    int target = staged_target(arg, source);
                    if (section_chunks[dat_index][
                            static_cast<std::size_t>(target)] != 1)
                        continue;

                    plan.stage_words[word_index] = word |
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
        plan.section_shared_bytes.push_back(section_shared_bytes);
    }
}

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

// Provide stable names for policy diagnostics and fallback reporting.
std::string_view
hier_smem_fallback_reason_name(HierSmemFallbackReason reason) {
    switch (reason) {
    case HierSmemFallbackReason::none:
        return "none";
    case HierSmemFallbackReason::disabled:
        return "disabled";
    case HierSmemFallbackReason::unvalidated_device:
        return "unvalidated_device";
    case HierSmemFallbackReason::not_staged:
        return "not_staged";
    case HierSmemFallbackReason::no_active_increment:
        return "no_active_increment";
    case HierSmemFallbackReason::incompatible_argument:
        return "incompatible_argument";
    case HierSmemFallbackReason::insufficient_shared_memory:
        return "insufficient_shared_memory";
    case HierSmemFallbackReason::low_compression:
        return "low_compression";
    case HierSmemFallbackReason::too_many_colours:
        return "too_many_colours";
    }

    return "unknown";
}

// Build the largest block-aligned plan that fits the shared-memory limit.
HierSmemPlanBuildResult build_hier_smem_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierSmemStagingDescriptor& descriptor,
    const HierSmemPlanOptions& options) {
    // Resolve runtime metadata once for every candidate.
    ResolvedInput resolved;
    auto reason = resolve_input(set, args, descriptor, options, true, resolved);
    if (reason != HierSmemFallbackReason::none)
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
        return {HierSmemFallbackReason::insufficient_shared_memory,
                std::nullopt};

    int chunk_size = static_cast<int>(std::min(
        static_cast<long long>(resolved.requested_chunk_size),
        blocks * options.block_size));

    HierSmemPlan plan;
    build_candidate(resolved, sections, chunk_size, options.exclusive_flush,
                    plan);
    plan.has_exclusive = plan.statistics.exclusive_owners > 0 ? 1 : 0;
    return {HierSmemFallbackReason::none, std::move(plan)};
}

// Colour each section's chunks, and each chunk's elements, over the targets of
// the coloured arguments.
HierSmemPlanBuildResult build_hier_colour_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierSmemStagingDescriptor& descriptor,
    const HierSmemPlanOptions& options) {
    ResolvedInput resolved;
    auto reason = resolve_input(set, args, descriptor, options, false, resolved);
    if (reason != HierSmemFallbackReason::none)
        return {reason, std::nullopt};

    assert(!sections.empty() && sections.front().start == 0);
    assert(sections.back().end == set->size + set->exec_size);

    // The active coloured arguments, each with its dat group.
    std::vector<std::pair<const op_arg *, std::size_t>> coloured;
    for (const auto& arg_desc : descriptor.args) {
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
    HierSmemPlan plan;
    plan.selected_chunk_size = chunk_size;
    plan.set_stride = resolved.set_stride;
    plan.thread_colours.assign(static_cast<std::size_t>(plan.set_stride), 0);
    plan.source_offsets.push_back(0);
    plan.section_chunk_offsets.push_back(0);
    plan.section_launch_offsets.push_back(0);
    plan.launch_chunk_offsets.push_back(0);

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
                    int target = staged_target(*arg, source);
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
                return {HierSmemFallbackReason::too_many_colours,
                        std::nullopt};

            for (int source = start; source < end; ++source)
                plan.thread_colours[static_cast<std::size_t>(source)] =
                    static_cast<std::uint8_t>(
                        colours[static_cast<std::size_t>(source - start)]);
            plan.chunk_thread_colours.push_back(thread_colours);
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
        int next = static_cast<int>(plan.chunk_order.size());
        for (int colour = 0; colour < block_colours; ++colour) {
            launch_next[static_cast<std::size_t>(colour)] = next;
            next += launch_sizes[static_cast<std::size_t>(colour)];
            plan.launch_chunk_offsets.push_back(next);
        }

        plan.chunk_order.resize(static_cast<std::size_t>(next));
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            auto colour = colours[static_cast<std::size_t>(chunk)];
            plan.chunk_order[static_cast<std::size_t>(
                launch_next[static_cast<std::size_t>(colour)]++)] =
                first_chunk + chunk;
        }

        plan.section_chunk_offsets.push_back(
            static_cast<int>(plan.num_chunks()));
        plan.section_launch_offsets.push_back(
            static_cast<int>(plan.launch_chunk_offsets.size()) - 1);
        plan.section_shared_bytes.push_back(0);
    }

    plan.statistics.launches = plan.launch_chunk_offsets.size() - 1;
    return {HierSmemFallbackReason::none, std::move(plan)};
}

void register_hier_smem_plan_owner(void *owner,
                                   HierSmemPlanReleaseCallback release) {
    assert(owner != nullptr && release != nullptr);
    std::scoped_lock lock(plan_owners_mutex);
    assert(std::none_of(plan_owners.begin(), plan_owners.end(),
                        [owner](const PlanOwner& item) {
                            return item.owner == owner;
                        }));
    plan_owners.push_back({owner, release});
}

void unregister_hier_smem_plan_owner(void *owner) {
    std::scoped_lock lock(plan_owners_mutex);
    std::erase_if(plan_owners, [owner](const PlanOwner& item) {
        return item.owner == owner;
    });
}

void release_hier_smem_plan_device_storage() {
    std::vector<PlanOwner> owners;
    {
        std::scoped_lock lock(plan_owners_mutex);
        owners = plan_owners;
    }

    for (const auto& owner : owners)
        owner.release(owner.owner);
}

} // namespace op::f2c
