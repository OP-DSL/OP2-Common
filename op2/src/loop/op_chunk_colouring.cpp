#include <loop/chunk_internal.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

namespace op::chunk::detail {
namespace {

// Chunks per wavefront window when the run does not set one.
constexpr int default_wave_window = 4096;

} // namespace

// Colour each section's chunks, and each chunk's elements, over the targets
// of the coloured arguments.  Each chunk applies its updates to its targets
// staged in scratch; chunk colouring keeps a batch's chunks off each other's
// targets, so every slot is seeded and stored back with plain loads and
// stores.
Result build_coloured(const op::loop::Configuration& configuration,
                      const op::loop::Sections& sections,
                      const ArgGroups& groups, const Constraints& constraints,
                      const Options& options) {
    ResolvedInput resolved;
    auto reason = resolve_input(configuration, groups, constraints, true,
                                resolved);
    if (!reason.empty())
        return {reason, nullptr};

    assert(sections.size() > 0 && sections[0].begin == 0);
    assert(sections[sections.size() - 1].end ==
           configuration.set->size + configuration.set->exec_size);

    // The active coloured arguments, each with its dat group.
    std::vector<std::pair<const op_arg *, std::size_t>> coloured;
    for (const ArgGroup& group : groups.args) {
        const op_arg& arg = configuration.args[static_cast<std::size_t>(group.arg)];
        if (arg.opt != 0)
            coloured.push_back({&arg, static_cast<std::size_t>(group.group)});
    }

    const std::size_t ndats = resolved.dats.size();
    std::vector<std::vector<std::uint64_t>> masks(ndats);
    for (std::size_t dat = 0; dat < ndats; ++dat)
        if (resolved.dats[dat].dat != nullptr)
            masks[dat].assign(
                static_cast<std::size_t>(resolved.dats[dat].target_extent), 0);

    int chunk_size = staged_chunk_size(resolved, constraints);
    if (chunk_size == 0)
        return {"insufficient_scratch", nullptr};

    // Every coloured argument is staged, so a chunk's distinct targets are its
    // staging's slot lists.
    auto partition = std::make_unique<Partition>();
    build_staging(resolved, sections, chunk_size, *partition, false);
    const auto& staging = partition->staging;
    auto chunk_targets = [&](int chunk, auto&& f) {
        for (std::size_t dat = 0; dat < ndats; ++dat) {
            auto region = static_cast<std::size_t>(chunk) * ndats + dat;
            auto first = static_cast<std::size_t>(staging.slot_offsets[region]);
            auto count = static_cast<std::size_t>(staging.stage_counts[region]);
            for (std::size_t s = first; s < first + count; ++s)
                f(dat, staging.slot_targets[s]);
        }
    };

    auto& colouring = partition->colouring.emplace();
    colouring.thread_colours.assign(
        static_cast<std::size_t>(partition->set_stride), 0);
    partition->section_batch_offsets.assign(1, 0);
    partition->batch_chunk_offsets.assign(1, 0);

    const bool wavefront = options.wavefront;
    const int wave_window =
        options.window > 0 ? options.window : default_wave_window;
    std::vector<std::vector<int>> latest(wavefront ? ndats : 0);
    if (wavefront) {
        colouring.wavefront = true;
        colouring.wave_pred_offsets.push_back(0);
        for (std::size_t dat = 0; dat < ndats; ++dat)
            latest[dat].assign(masks[dat].size(), -1);
    }

    std::vector<int> colours;

    for (int section = 0; section < sections.size(); ++section) {
        const int first_chunk =
            partition->section_chunk_offsets[static_cast<std::size_t>(section)];
        const int last_chunk =
            partition->section_chunk_offsets[static_cast<std::size_t>(section) + 1];

        for (int chunk = first_chunk; chunk < last_chunk; ++chunk) {
            int start = partition->source_offsets[static_cast<std::size_t>(chunk)];
            int end = partition->source_offsets[static_cast<std::size_t>(chunk) + 1];

            int thread_colours = greedy_colour(
                end - start,
                [&](int element, auto&& f) {
                    for (const auto& [arg, dat] : coloured)
                        f(dat, arg_target(*arg, start + element));
                },
                masks, colours);
            if (thread_colours > 255)
                return {"too_many_colours", nullptr};

            for (int source = start; source < end; ++source)
                colouring.thread_colours[static_cast<std::size_t>(source)] =
                    static_cast<std::uint8_t>(
                        colours[static_cast<std::size_t>(source - start)]);
            colouring.chunk_thread_colours.push_back(thread_colours);
            partition->statistics.max_thread_colours =
                std::max(partition->statistics.max_thread_colours, thread_colours);
        }

        // Colour the section's chunks, then list them batch by batch.  A
        // wavefront partition colours and lists them window by window
        // instead, and runs the section as one batch.
        const int nsection = last_chunk - first_chunk;
        const int window = wavefront ? wave_window : std::max(nsection, 1);
        const int section_first = static_cast<int>(colouring.chunk_order.size());
        for (int window_first = first_chunk; window_first < last_chunk;
             window_first += window) {
            const int nchunks = std::min(window, last_chunk - window_first);
            int chunk_colours = greedy_colour(
                nchunks,
                [&](int chunk, auto&& f) { chunk_targets(window_first + chunk, f); },
                masks, colours);

            std::vector<int> batch_sizes(static_cast<std::size_t>(chunk_colours), 0);
            for (int chunk = 0; chunk < nchunks; ++chunk)
                ++batch_sizes[static_cast<std::size_t>(
                    colours[static_cast<std::size_t>(chunk)])];

            std::vector<int> batch_next(static_cast<std::size_t>(chunk_colours));
            int next = static_cast<int>(colouring.chunk_order.size());
            for (int colour = 0; colour < chunk_colours; ++colour) {
                batch_next[static_cast<std::size_t>(colour)] = next;
                next += batch_sizes[static_cast<std::size_t>(colour)];
                if (!wavefront)
                    partition->batch_chunk_offsets.push_back(next);
            }

            colouring.chunk_order.resize(static_cast<std::size_t>(next));
            for (int chunk = 0; chunk < nchunks; ++chunk) {
                auto colour = colours[static_cast<std::size_t>(chunk)];
                colouring.chunk_order[static_cast<std::size_t>(
                    batch_next[static_cast<std::size_t>(colour)]++)] =
                    window_first + chunk;
            }
        }

        if (wavefront) {
            partition->batch_chunk_offsets.push_back(
                static_cast<int>(colouring.chunk_order.size()));

            // A position's predecessors are the latest earlier position
            // touching each of its targets; those are ordered among
            // themselves, so waiting on them covers every earlier one.
            for (int position = section_first;
                 position < static_cast<int>(colouring.chunk_order.size());
                 ++position) {
                auto first_pred = colouring.wave_preds.size();
                chunk_targets(
                    colouring.chunk_order[static_cast<std::size_t>(position)],
                    [&](std::size_t dat, int target) {
                        int& last = latest[dat][static_cast<std::size_t>(target)];
                        if (last >= 0 &&
                            std::find(colouring.wave_preds.begin() + first_pred,
                                      colouring.wave_preds.end(), last) ==
                                colouring.wave_preds.end())
                            colouring.wave_preds.push_back(last);
                        last = position;
                    });
                colouring.wave_pred_offsets.push_back(
                    static_cast<int>(colouring.wave_preds.size()));
            }
            for (int chunk = first_chunk; chunk < last_chunk; ++chunk)
                chunk_targets(chunk, [&](std::size_t dat, int target) {
                    latest[dat][static_cast<std::size_t>(target)] = -1;
                });
        }

        partition->section_batch_offsets.push_back(
            static_cast<int>(partition->batch_chunk_offsets.size()) - 1);
    }

    size_batches(*partition);
    return {{}, std::move(partition)};
}

} // namespace op::chunk::detail
