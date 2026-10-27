#include <plan/chunks.h>

#include <plan/greedy_colour.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace op::plan {

using detail::arg_target;
using detail::greedy_colour;

// --- References -----------------------------------------------------------------

std::string_view updated_references(std::span<const op_arg> args,
                                    References& references) {
    references = {};
    std::vector<op_dat> updated;
    for (const op_arg& arg : args)
        if (arg.opt != 0 && arg.argtype == OP_ARG_DAT &&
            arg.map != nullptr && (arg.acc == OP_INC || arg.acc == OP_RW) &&
            std::find(updated.begin(), updated.end(), arg.dat) == updated.end())
            updated.push_back(arg.dat);
    if (updated.empty())
        return "no_active_argument";

    // Every indirect argument on an updated dat, its reads too, so no chunk
    // reads a target another chunk of its batch updates.
    std::vector<op_dat> dats;
    for (const op_arg& arg : args) {
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT || arg.map == nullptr ||
            std::find(updated.begin(), updated.end(), arg.dat) == updated.end())
            continue;
        if (arg.idx < 0)
            return "incompatible_argument";

        auto found = std::find(dats.begin(), dats.end(), arg.dat);
        if (found == dats.end()) {
            dats.push_back(arg.dat);
            references.groups.push_back(Group::of(arg));
            found = dats.end() - 1;
        }
        references.items.push_back({&arg, static_cast<int>(found - dats.begin())});
    }
    return {};
}

// --- Chunks ---------------------------------------------------------------------

Chunks chunk_sections(const op::loop::Sections& sections, int size) {
    assert(size > 0);
    Chunks chunks;
    chunks.size = size;
    chunks.offsets.push_back(sections.size() > 0 ? sections[0].begin : 0);
    chunks.section_offsets.push_back(0);
    for (const op::loop::Section& section : sections.span()) {
        assert(section.begin == chunks.offsets.back());
        for (int start = section.begin; start < section.end;) {
            int end = static_cast<int>(std::min(static_cast<long long>(section.end),
                                                static_cast<long long>(start) + size));
            chunks.offsets.push_back(end);
            start = end;
        }
        chunks.section_offsets.push_back(chunks.count());
    }
    return chunks;
}

ChunkTargets chunk_targets(const Chunks& chunks, const References& references) {
    ChunkTargets targets;
    targets.groups = static_cast<int>(references.groups.size());
    for (const Group& group : references.groups)
        targets.extents.push_back(group.extent);
    targets.offsets.reserve(static_cast<std::size_t>(chunks.count()) *
                                references.groups.size() + 1);

    // Per group, each target's references in the chunk, and the chunk's
    // distinct targets.
    std::vector<std::vector<int>> counts(references.groups.size());
    std::vector<std::vector<int>> touched(references.groups.size());
    for (std::size_t group = 0; group < counts.size(); ++group)
        counts[group].assign(static_cast<std::size_t>(references.groups[group].extent), 0);

    for (int chunk = 0; chunk < chunks.count(); ++chunk) {
        for (int element = chunks.begin(chunk); element < chunks.end(chunk); ++element)
            for (const Reference& reference : references.items) {
                if (reference.arg->opt == 0)
                    continue;
                auto group = static_cast<std::size_t>(reference.group);
                int target = arg_target(*reference.arg, element);
                assert(target >= 0 && target < references.groups[group].extent);
                int& count = counts[group][static_cast<std::size_t>(target)];
                if (count++ == 0)
                    touched[group].push_back(target);
                ++targets.total_references;
            }

        for (std::size_t group = 0; group < touched.size(); ++group) {
            auto& list = touched[group];
            std::sort(list.begin(), list.end());
            targets.offsets.push_back(static_cast<int>(targets.targets.size()));
            for (int target : list) {
                int& count = counts[group][static_cast<std::size_t>(target)];
                targets.targets.push_back(target);
                targets.references.push_back(count);
                count = 0;
            }
            list.clear();
        }
    }
    targets.offsets.push_back(static_cast<int>(targets.targets.size()));
    return targets;
}

// --- Batches --------------------------------------------------------------------

ChunkBatches section_batches(const Chunks& chunks) {
    ChunkBatches batches;
    batches.batch_offsets.push_back(0);
    batches.section_batches.push_back(0);
    for (std::size_t section = 0; section + 1 < chunks.section_offsets.size(); ++section) {
        if (chunks.section_offsets[section + 1] > chunks.section_offsets[section])
            batches.batch_offsets.push_back(chunks.section_offsets[section + 1]);
        batches.section_batches.push_back(batches.count());
    }
    return batches;
}

ChunkBatches colour_chunks(const Chunks& chunks, const ChunkTargets& targets,
                           int window) {
    std::vector<std::vector<std::uint64_t>> masks(targets.extents.size());
    for (std::size_t group = 0; group < masks.size(); ++group)
        masks[group].assign(static_cast<std::size_t>(targets.extents[group]), 0);

    ChunkBatches batches;
    batches.order.reserve(static_cast<std::size_t>(chunks.count()));
    batches.batch_offsets.push_back(0);
    batches.section_batches.push_back(0);
    std::vector<int> colours;

    const auto sections = static_cast<int>(chunks.section_offsets.size()) - 1;
    for (int section = 0; section < sections; ++section) {
        const int first = chunks.section_offsets[static_cast<std::size_t>(section)];
        const int last = chunks.section_offsets[static_cast<std::size_t>(section) + 1];
        const int step = window > 0 ? window : std::max(last - first, 1);

        // Colour a window's chunks, then list them batch by batch.
        for (int window_first = first; window_first < last; window_first += step) {
            const int count = std::min(step, last - window_first);
            int ncolours = greedy_colour(
                count,
                [&](int chunk, auto&& f) { targets.for_each(window_first + chunk, f); },
                masks, colours);

            std::vector<int> next(static_cast<std::size_t>(ncolours), 0);
            for (int chunk = 0; chunk < count; ++chunk)
                ++next[static_cast<std::size_t>(colours[static_cast<std::size_t>(chunk)])];
            int position = static_cast<int>(batches.order.size());
            for (int& start : next) {
                int size = start;
                start = position;
                position += size;
                batches.batch_offsets.push_back(position);
            }

            batches.order.resize(static_cast<std::size_t>(position));
            for (int chunk = 0; chunk < count; ++chunk)
                batches.order[static_cast<std::size_t>(
                    next[static_cast<std::size_t>(colours[static_cast<std::size_t>(chunk)])]++)] =
                    window_first + chunk;
        }
        batches.section_batches.push_back(batches.count());
    }
    return batches;
}

Wavefront wavefront(const ChunkBatches& coloured, const ChunkTargets& targets) {
    Wavefront wave;
    wave.batches.order = coloured.order;
    wave.batches.batch_offsets.push_back(0);
    wave.batches.section_batches.push_back(0);
    wave.pred_offsets.push_back(0);

    // The latest position so far reaching each target, within the section.
    std::vector<std::vector<int>> latest(targets.extents.size());
    for (std::size_t group = 0; group < latest.size(); ++group)
        latest[group].assign(static_cast<std::size_t>(targets.extents[group]), -1);

    const auto sections = static_cast<int>(coloured.section_batches.size()) - 1;
    for (int section = 0; section < sections; ++section) {
        int begin = wave.batches.batch_offsets.back();
        int end = begin;
        for (int batch = 0; batch < coloured.batches(section); ++batch)
            end = coloured.positions(section, batch).second;
        if (end > begin)
            wave.batches.batch_offsets.push_back(end);
        wave.batches.section_batches.push_back(wave.batches.count());

        // Positions sharing a target are ordered among themselves, so waiting
        // on the latest earlier one covers every earlier one.
        for (int position = begin; position < end; ++position) {
            auto first = wave.preds.size();
            targets.for_each(coloured.chunk(position), [&](std::size_t group, int target) {
                int& last = latest[group][static_cast<std::size_t>(target)];
                if (last >= 0 && std::find(wave.preds.begin() + static_cast<std::ptrdiff_t>(first),
                                           wave.preds.end(), last) == wave.preds.end())
                    wave.preds.push_back(last);
                last = position;
            });
            wave.pred_offsets.push_back(static_cast<int>(wave.preds.size()));
        }
        for (int position = begin; position < end; ++position)
            targets.for_each(coloured.chunk(position), [&](std::size_t group, int target) {
                latest[group][static_cast<std::size_t>(target)] = -1;
            });
    }
    return wave;
}

// --- Element colours ------------------------------------------------------------

ElementColours colour_elements(const Chunks& chunks, const References& references) {
    std::vector<std::vector<std::uint64_t>> masks(references.groups.size());
    for (std::size_t group = 0; group < masks.size(); ++group)
        masks[group].assign(static_cast<std::size_t>(references.groups[group].extent), 0);

    ElementColours result;
    result.colours.assign(chunks.offsets.empty()
                              ? 0
                              : static_cast<std::size_t>(chunks.offsets.back()),
                          0);
    std::vector<int> colours;
    for (int chunk = 0; chunk < chunks.count(); ++chunk) {
        const int start = chunks.begin(chunk);
        int count = greedy_colour(
            chunks.end(chunk) - start,
            [&](int element, auto&& f) {
                for (const Reference& reference : references.items)
                    if (reference.arg->opt != 0)
                        f(static_cast<std::size_t>(reference.group),
                          arg_target(*reference.arg, start + element));
            },
            masks, colours);
        std::copy(colours.begin(), colours.end(),
                  result.colours.begin() + start);
        result.counts.push_back(count);
        result.max = std::max(result.max, count);
    }
    return result;
}

} // namespace op::plan
