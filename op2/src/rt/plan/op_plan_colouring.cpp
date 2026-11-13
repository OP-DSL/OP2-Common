#include <op2/rt/plan/colouring.h>

#include <op2/rt/plan/greedy_colour.h>

#include <cassert>
#include <cstdint>
#include <numeric>
#include <utility>

namespace op::rt::plan {

SetColouring colour_set(std::span<const op_arg> args, const op::rt::loop::Sections& sections) {
    assert(sections.size() > 0);

    // The elements the sections run.
    const int exec_length = sections[sections.size() - 1].end;

    // The updated targets, one mask vector per updated dat.
    std::vector<std::pair<const op_arg *, std::size_t>> updates;
    std::vector<const void *> dats;
    std::vector<op_set> sets;
    for (const op_arg& arg : args) {
        if (arg.opt == 0 || !loop::has_target(arg) || arg.idx < 0 ||
            (arg.acc != OP_INC && arg.acc != OP_RW))
            continue;

        std::size_t dat = 0;
        while (dat < dats.size() && dats[dat] != loop::target(arg))
            ++dat;
        if (dat == dats.size()) {
            dats.push_back(loop::target(arg));
            sets.push_back(loop::target_set(arg));
        }
        updates.push_back({&arg, dat});
    }

    std::vector<std::vector<std::uint64_t>> masks(dats.size());
    for (std::size_t dat = 0; dat < dats.size(); ++dat) {
        op_set target_set = sets[dat];
        masks[dat].assign(static_cast<std::size_t>(target_set->size +
                                                   target_set->exec_size +
                                                   target_set->nonexec_size),
                          0);
    }

    SetColouring colouring;
    colouring.order.resize(static_cast<std::size_t>(exec_length));
    std::vector<int> colours(static_cast<std::size_t>(exec_length), -1);

    // Colour [begin, end) with colours from base, returning the count used.
    auto colour_range = [&](int begin, int end, int base) {
        std::vector<int> range_colours;
        int count = greedy_colour(
            end - begin,
            [&](int element, auto&& f) {
                for (const auto& [arg, dat] : updates)
                    f(dat, arg_target(*arg, begin + element));
            },
            masks, range_colours);
        for (int element = begin; element < end; ++element)
            colours[static_cast<std::size_t>(element)] =
                base + range_colours[static_cast<std::size_t>(element - begin)];
        return count;
    };

    // A section at a time, so no colour holds elements of two.
    auto phases = sections.phases();
    for (int section = 0; section < sections.size(); ++section) {
        if (section == phases.wait_before)
            colouring.core_colours = colouring.colours;
        colouring.colours += colour_range(sections[section].begin, sections[section].end,
                                          colouring.colours);
        if (section == phases.globals_after)
            colouring.owned_colours = colouring.colours;
    }
    if (phases.wait_before >= sections.size())
        colouring.core_colours = colouring.colours;

    // Each colour's elements, in element order.
    colouring.offsets.assign(static_cast<std::size_t>(colouring.colours) + 1, 0);
    for (int colour : colours)
        ++colouring.offsets[static_cast<std::size_t>(colour) + 1];
    std::partial_sum(colouring.offsets.begin(), colouring.offsets.end(),
                     colouring.offsets.begin());
    std::vector<int> next(colouring.offsets.begin(), colouring.offsets.end() - 1);
    for (int element = 0; element < exec_length; ++element)
        colouring.order[static_cast<std::size_t>(
            next[static_cast<std::size_t>(colours[static_cast<std::size_t>(element)])]++)] =
            element;

    return colouring;
}

} // namespace op::rt::plan
