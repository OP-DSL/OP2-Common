#include <loop/chunk_internal.h>

#include <op_lib_c.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <utility>

namespace op::chunk {

// --- Options ------------------------------------------------------------------

namespace {

Options run_options;

} // namespace

const Options& options() { return run_options; }

void init_options(int default_variants) {
    static bool initialized = false;
    if (initialized)
        return;
    initialized = true;
    run_options.variants = default_variants;

    op::loop::env::parse_number("OP_CHUNK_ATOMICS_MIN_COMPRESSION",
                                run_options.min_compression);
    op::loop::env::parse_number("OP_CHUNK_ATOMICS_CHUNK_SIZE",
                                run_options.atomics_chunk_size);
    op::loop::env::parse_number("OP_CHUNK_COLOURING_CHUNK_SIZE",
                                run_options.colouring_chunk_size);
    op::loop::env::parse_number("OP_CHUNK_COLOURING_WINDOW", run_options.window);

    const std::pair<const char *, int> bits[] = {
        {"OP_CHUNK_ATOMICS_UNSTAGED_SINGLES", variant_unstaged_singles},
        {"OP_CHUNK_COLOURING_WAVEFRONT", variant_wavefront},
    };
    for (const auto& [variable, bit] : bits) {
        bool on = (run_options.variants & bit) != 0;
        if (!op::loop::env::parse_bool(variable, on))
            continue;
        run_options.variants = on ? (run_options.variants | bit)
                                  : (run_options.variants & ~bit);
        std::printf("%s %s\n", on ? "Enabling" : "Disabling", variable);
    }
}

// --- Resolution ---------------------------------------------------------------

namespace detail {

std::string_view resolve_input(const op::loop::Configuration& configuration,
                               const ArgGroups& groups,
                               const Constraints& constraints,
                               int requested_chunk_size, bool read_write,
                               ResolvedInput& resolved) {
    // The groups are generated alongside the kernel they describe, so their
    // shape is an internal invariant rather than a runtime condition.
    op_set set = configuration.set;
    std::span<const op_arg> args = configuration.args;
    assert(set != nullptr);
    assert(!groups.args.empty() && groups.groups > 0);
    assert(groups.args.size() <= args.size());
    assert(constraints.lanes > 0);

    resolved.set_stride = op::loop::indirect_set_stride(set);
    resolved.args = args;
    resolved.groups = groups.args;
    resolved.dats.assign(static_cast<std::size_t>(groups.groups), {});

    // Round the requested chunk size up to a lanes multiple, clamped so the
    // result stays an int.
    int requested = requested_chunk_size > 0 ? requested_chunk_size
                                             : constraints.lanes;
    long long blocks = (static_cast<long long>(requested) +
                        constraints.lanes - 1) / constraints.lanes;
    long long limit = std::numeric_limits<int>::max() / constraints.lanes;
    resolved.chunk_size = static_cast<int>(
        std::max(1LL, std::min(blocks, limit)) * constraints.lanes);

    // Resolve optional state and the runtime identities behind each group.
    std::vector<bool> grouped_args(args.size(), false);
    bool any_active = false;
    for (const ArgGroup& group : groups.args) {
        assert(group.arg >= 0 && group.group >= 0);
        const auto arg_index = static_cast<std::size_t>(group.arg);
        const auto dat_index = static_cast<std::size_t>(group.group);
        assert(arg_index < args.size() && dat_index < resolved.dats.size());
        assert(!grouped_args[arg_index]);
        grouped_args[arg_index] = true;

        const op_arg& arg = args[arg_index];
        if (arg.opt == 0)
            continue;

        assert(arg.argtype == OP_ARG_DAT);
        assert(read_write || arg.acc == OP_INC);
        assert(arg.dat != nullptr && arg.map != nullptr &&
               arg.map_data != nullptr);
        assert(arg.idx >= 0 && arg.idx < arg.map->dim);
        assert(arg.dim > 0 && arg.size % arg.dim == 0);

        any_active = true;
        auto& resolved_dat = resolved.dats[dat_index];
        op_set target_set = arg.dat->set;

        if (resolved_dat.dat == nullptr) {
            resolved_dat.dat = arg.dat;
            resolved_dat.dimension = arg.dim;
            resolved_dat.scalar_size =
                static_cast<std::size_t>(arg.size / arg.dim);
            resolved_dat.target_extent = target_set->size +
                                         target_set->exec_size +
                                         target_set->nonexec_size;
        } else if (resolved_dat.dat != arg.dat) {
            return "incompatible_argument";
        }
    }

    if (!any_active)
        return "no_active_argument";

    // The translator groups dat arguments by source expression, so two groups
    // can still resolve to one runtime op_dat (the same dat passed through two
    // dummy arguments, say).  Each group is planned from its own targets: a
    // staged group owns an independent region, and a coloured group's
    // conflicts are found only within it, so two groups on one dat could race
    // on the same global address.
    //
    // An ungrouped argument on a grouped dat is a related hazard.  Colouring
    // does not order it, and under staging its reads would miss increments
    // still sitting in scratch.  That loop is already unsound under global
    // atomics, but staging turns "possibly stale" into "certainly stale", so
    // reject rather than change its behaviour.
    //
    // One pass answers both: every dat argument must map to at most one group.
    std::unordered_map<const op_dat_core *, std::size_t> claimed;
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        const auto *dat = resolved.dats[dat_index].dat;
        if (dat != nullptr && !claimed.emplace(dat, dat_index).second)
            return "incompatible_argument";
    }

    // An ungrouped indirect update of any dat is one the strategy would not
    // order at all, so it is rejected too.
    for (std::size_t arg_index = 0; arg_index < args.size(); ++arg_index) {
        const op_arg& arg = args[arg_index];
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT || arg.dat == nullptr ||
            grouped_args[arg_index])
            continue;

        bool update = arg.acc == OP_INC || (read_write && arg.acc == OP_RW);
        if (claimed.count(arg.dat) != 0 || (update && arg.idx >= 0))
            return "incompatible_argument";
    }

    return {};
}

// --- Staging --------------------------------------------------------------------

namespace {

// The padding the region layout can insert.  The first region starts at
// offset zero, so only the regions after it can need aligning.
std::size_t alignment_slack(const ResolvedInput& resolved) {
    std::size_t slack = 0;
    for (std::size_t dat_index = 1; dat_index < resolved.dats.size();
         ++dat_index)
        slack += resolved.dats[dat_index].scalar_size > 0
                     ? resolved.dats[dat_index].scalar_size - 1
                     : 0;
    return slack;
}

// The active arguments on each staged dat.
std::vector<int> args_per_dat(const ResolvedInput& resolved) {
    std::vector<int> counts(resolved.dats.size(), 0);
    for (const ArgGroup& group : resolved.groups)
        if (resolved.args[static_cast<std::size_t>(group.arg)].opt != 0)
            ++counts[static_cast<std::size_t>(group.group)];
    return counts;
}

// The most scratch bytes a single source element can require, which is what
// bounds the chunk size before any map is inspected.
std::size_t bytes_per_source_element(const ResolvedInput& resolved) {
    auto counts = args_per_dat(resolved);
    std::size_t total = 0;
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        const auto& dat = resolved.dats[dat_index];
        if (dat.dat == nullptr)
            continue;
        // Each active argument can reach one distinct target per element.
        total += static_cast<std::size_t>(counts[dat_index]) *
                 static_cast<std::size_t>(dat.dimension) * dat.scalar_size;
    }
    return total;
}

} // namespace

std::size_t chunk_scratch_bytes(const ResolvedInput& resolved,
                                std::span<const int> counts) {
    std::size_t bytes = 0;
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index) {
        // Inactive groups keep their place with a zero-length region, so the
        // staged kernel can walk this layout without branching on opt state.
        const auto& dat = resolved.dats[dat_index];
        if (dat.dat == nullptr)
            continue;
        std::size_t alignment = dat.scalar_size;
        bytes = (bytes + alignment - 1) & ~(alignment - 1);
        bytes += static_cast<std::size_t>(counts[dat_index]) *
                 static_cast<std::size_t>(dat.dimension) * dat.scalar_size;
    }
    return bytes;
}

int staged_chunk_size(const ResolvedInput& resolved,
                      const Constraints& constraints) {
    // A chunk's distinct targets cannot outnumber its source references, so
    // the exact partition is bounded by chunk_size * bytes_per_source_element.
    // Solving that for chunk_size picks a size guaranteed to fit up front, and
    // the partition is then built exactly once.  Region alignment adds at most
    // one scalar of padding per staged dat, which comes off the limit first.
    std::size_t slack = alignment_slack(resolved);
    std::size_t limit = constraints.scratch_bytes > slack
                            ? constraints.scratch_bytes - slack
                            : 0;

    std::size_t per_element = bytes_per_source_element(resolved);
    assert(per_element > 0);

    // A region's slots must also fit the stage word.
    auto counts = args_per_dat(resolved);
    int references = std::max(*std::max_element(counts.begin(), counts.end()),
                              1);
    auto capacity = std::min(static_cast<long long>(limit / per_element),
                             static_cast<long long>(constraints.max_slots /
                                                    references));
    long long blocks = capacity / constraints.lanes;
    if (blocks < 1)
        return 0;

    return static_cast<int>(std::min(
        static_cast<long long>(resolved.chunk_size),
        blocks * constraints.lanes));
}

void build_staging(const ResolvedInput& resolved,
                   const op::loop::Sections& sections, int chunk_size,
                   Partition& partition, bool unstaged_singles) {
    partition = {};
    partition.chunk_size = chunk_size;
    partition.set_stride = resolved.set_stride;
    auto& staging = partition.staging;

    std::size_t num_chunks = 0;
    for (const auto& section : sections.span())
        num_chunks += static_cast<std::size_t>(
            (static_cast<long long>(section.size()) + chunk_size - 1) /
            chunk_size);

    staging.stage_words.assign(
        resolved.groups.size() * static_cast<std::size_t>(partition.set_stride),
        0);
    staging.stage_counts.reserve(num_chunks * resolved.dats.size());
    staging.slot_offsets.reserve(num_chunks * resolved.dats.size());
    staging.chunk_scratch_bytes.reserve(num_chunks);
    partition.source_offsets.reserve(num_chunks + 1);
    partition.section_chunk_offsets.reserve(
        static_cast<std::size_t>(sections.size()) + 1);

    // One reusable map per active dat from each target to its references in
    // the chunk, then to its slot, and each chunk's distinct targets.
    std::vector<std::vector<int>> inverse(resolved.dats.size());
    std::vector<std::vector<int>> touched(resolved.dats.size());
    for (std::size_t dat_index = 0; dat_index < resolved.dats.size();
         ++dat_index)
        if (resolved.dats[dat_index].dat != nullptr)
            inverse[dat_index].assign(
                static_cast<std::size_t>(resolved.dats[dat_index].target_extent),
                -1);

    // Visit every active staged reference of the sources [start, end).
    auto for_each_reference = [&](int start, int end, auto&& visit) {
        for (int source = start; source < end; ++source) {
            for (std::size_t staged_arg = 0; staged_arg < resolved.groups.size();
                 ++staged_arg) {
                const ArgGroup& group = resolved.groups[staged_arg];
                const op_arg& arg = resolved.args[static_cast<std::size_t>(group.arg)];
                if (arg.opt == 0)
                    continue;

                const auto dat_index = static_cast<std::size_t>(group.group);
                int target = arg_target(arg, source);
                assert(target >= 0 &&
                       target < resolved.dats[dat_index].target_extent);
                visit(staging.stage_words[
                          staged_arg * static_cast<std::size_t>(partition.set_stride) +
                          static_cast<std::size_t>(source)],
                      dat_index, target);
            }
        }
    };

    partition.source_offsets.push_back(0);
    partition.section_chunk_offsets.push_back(0);

    for (const op::loop::Section& section : sections.span()) {
        // Chunk each section independently so batches retain their halo-wait
        // and global-processing boundaries.
        for (int start = section.begin; start < section.end;) {
            int end = static_cast<int>(std::min(
                static_cast<long long>(section.end),
                static_cast<long long>(start) + chunk_size));

            // Count the chunk's references to each distinct target.
            for_each_reference(start, end,
                               [&](StageWord&, std::size_t dat_index, int target) {
                int& references = inverse[dat_index][static_cast<std::size_t>(target)];
                if (references < 0) {
                    references = 0;
                    touched[dat_index].push_back(target);
                }
                ++references;
                ++partition.statistics.raw_references;
            });

            // Give the targets slots in target order, as op_plan's ind_map
            // has them, so a lane per slot seeds and flushes ascending
            // addresses.  With unstaged_singles a target referenced once stays
            // in global memory: staging it would save no atomic.
            for (std::size_t dat_index = 0; dat_index < touched.size();
                 ++dat_index) {
                auto& targets = touched[dat_index];
                std::sort(targets.begin(), targets.end());
                staging.slot_offsets.push_back(
                    static_cast<int>(staging.slot_targets.size()));

                int slots = 0;
                for (int target : targets) {
                    int& entry = inverse[dat_index][static_cast<std::size_t>(target)];
                    if (unstaged_singles && entry == 1) {
                        entry = unstaged_word;
                        ++partition.statistics.unstaged_singles;
                        continue;
                    }
                    entry = slots++;
                    staging.slot_targets.push_back(target);
                }
                staging.stage_counts.push_back(slots);
                partition.statistics.distinct_targets += targets.size();
            }

            for_each_reference(start, end,
                               [&](StageWord& word, std::size_t dat_index,
                                   int target) {
                word = static_cast<StageWord>(
                    inverse[dat_index][static_cast<std::size_t>(target)]);
            });

            staging.chunk_scratch_bytes.push_back(chunk_scratch_bytes(
                resolved,
                std::span<const int>{staging.stage_counts}.last(touched.size())));

            // Reset only the map entries this chunk touched.
            for (std::size_t dat_index = 0; dat_index < touched.size();
                 ++dat_index) {
                for (int target : touched[dat_index])
                    inverse[dat_index][static_cast<std::size_t>(target)] = -1;
                touched[dat_index].clear();
            }

            partition.source_offsets.push_back(end);
            start = end;
        }

        partition.section_chunk_offsets.push_back(
            static_cast<int>(partition.num_chunks()));
    }

    // One batch per section until a colouring says otherwise.
    for (int section = 0; section <= sections.size(); ++section)
        partition.section_batch_offsets.push_back(section);
    partition.batch_chunk_offsets = partition.section_chunk_offsets;
}

void size_batches(Partition& partition) {
    partition.batch_scratch_bytes.clear();
    for (std::size_t batch = 0; batch + 1 < partition.batch_chunk_offsets.size();
         ++batch)
        partition.batch_scratch_bytes.push_back(partition.max_chunk_scratch_bytes(
            partition.batch_chunk_offsets[batch],
            partition.batch_chunk_offsets[batch + 1]));
    partition.statistics.batches = partition.batch_scratch_bytes.size();
}

// --- The staged kind -------------------------------------------------------------

namespace {

// The largest lanes-aligned staging partition that fits the scratch.
// Concurrent chunks of a section may share targets, so every slot flushes
// with a global atomic.
Result build_staged(const op::loop::Configuration& configuration,
                    const op::loop::Sections& sections, const ArgGroups& groups,
                    const Constraints& constraints, const Options& options) {
    ResolvedInput resolved;
    int requested = constraints.elements > 0 ? constraints.elements
                    : options.atomics_chunk_size > 0 ? options.atomics_chunk_size
                                                     : OP_part_size;
    auto reason = resolve_input(configuration, groups, constraints, requested,
                                false, resolved);
    if (!reason.empty())
        return {reason, nullptr};

    assert(sections.size() > 0 && sections[0].begin == 0);
    assert(sections[sections.size() - 1].end ==
           configuration.set->size + configuration.set->exec_size);

    int chunk_size = staged_chunk_size(resolved, constraints);
    if (chunk_size == 0)
        return {"insufficient_scratch", nullptr};

    bool unstaged_singles = (options.variants & variant_unstaged_singles) != 0;
    auto partition = std::make_unique<Partition>();
    build_staging(resolved, sections, chunk_size, *partition, unstaged_singles);
    size_batches(*partition);

    // A partition that barely combines references is rejected.  With unstaged
    // singles it costs little more than global atomics, since its single
    // references are those.
    if (!unstaged_singles &&
        partition->statistics.compression() < options.min_compression)
        return {"low_compression", std::move(partition)};

    return {{}, std::move(partition)};
}

} // namespace
} // namespace detail

Result build_partition(Kind kind, const op::loop::Configuration& configuration,
                       const op::loop::Sections& sections,
                       const ArgGroups& groups, const Constraints& constraints,
                       const Options& options) {
    if (kind == Kind::staged)
        return detail::build_staged(configuration, sections, groups, constraints,
                                    options);
    return detail::build_coloured(configuration, sections, groups, constraints,
                                  options);
}

} // namespace op::chunk
