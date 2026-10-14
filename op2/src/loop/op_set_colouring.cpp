#include <loop/set_colouring.h>

#include <loop/greedy_colour.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <utility>

namespace op::colouring {

SetColouring colour_set(op_set set, std::span<const op_arg> args,
                        bool separate_owned) {
    assert(set != nullptr);

    // The elements run: the exec halo too when an indirect argument updates.
    int exec_length = set->size;
    for (const op_arg& arg : args)
        if (arg.opt != 0 && arg.argtype == OP_ARG_DAT && arg.idx >= 0 &&
            arg.acc != OP_READ) {
            exec_length += set->exec_size;
            break;
        }

    // The updated targets, one mask vector per updated dat.
    std::vector<std::pair<const op_arg *, std::size_t>> updates;
    std::vector<op_dat> dats;
    for (const op_arg& arg : args) {
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT || arg.idx < 0 ||
            (arg.acc != OP_INC && arg.acc != OP_RW))
            continue;

        std::size_t dat = 0;
        while (dat < dats.size() && dats[dat] != arg.dat)
            ++dat;
        if (dat == dats.size())
            dats.push_back(arg.dat);
        updates.push_back({&arg, dat});
    }

    std::vector<std::vector<std::uint64_t>> masks(dats.size());
    for (std::size_t dat = 0; dat < dats.size(); ++dat) {
        op_set target_set = dats[dat]->set;
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
        int count = op::loop::detail::greedy_colour(
            end - begin,
            [&](int element, auto&& f) {
                for (const auto& [arg, dat] : updates)
                    f(dat, op::loop::detail::arg_target(*arg, begin + element));
            },
            masks, range_colours);
        for (int element = begin; element < end; ++element)
            colours[static_cast<std::size_t>(element)] =
                base + range_colours[static_cast<std::size_t>(element - begin)];
        return count;
    };

    const int owned = std::min(set->size, exec_length);
    if (separate_owned) {
        colouring.owned_colours = colour_range(0, owned, 0);
        colouring.colours =
            colouring.owned_colours +
            colour_range(owned, exec_length, colouring.owned_colours);
    } else {
        colouring.colours = colour_range(0, exec_length, 0);
        colouring.owned_colours = colouring.colours;
    }

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

} // namespace op::colouring
