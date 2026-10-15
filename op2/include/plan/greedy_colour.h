#pragma once

// Internal to the planners: the greedy colouring the set and chunk
// colourings share.

#include <op_lib_core.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace op::plan::detail {

// The global target an argument reaches from one source element.
inline int arg_target(const op_arg& arg, int source) {
    return arg.map_data[static_cast<std::size_t>(source) *
                            static_cast<std::size_t>(arg.map->dim) +
                        static_cast<std::size_t>(arg.idx)];
}

// One target of a coloured item: a dat group and an element of its set.
struct ColourTarget {
    std::size_t dat;
    int target;
};

// Colour items greedily so that no two items of a colour share a target,
// trying 64 colours a round.  targets(i, f) calls f(dat, target) for each of
// item i's targets, and masks, one per target of each dat, start and end
// clear.  Return the colour count.
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
            targets(item, [&](std::size_t dat, int target) {
                used |= masks[dat][static_cast<std::size_t>(target)];
            });
            if (used == ~std::uint64_t{0})
                continue;

            int bit = std::countr_one(used);
            colour = base + bit;
            ncolours = std::max(ncolours, colour + 1);
            ++coloured;

            targets(item, [&](std::size_t dat, int target) {
                auto& mask = masks[dat][static_cast<std::size_t>(target)];
                if (mask == 0)
                    marked.push_back({dat, target});
                mask |= std::uint64_t{1} << bit;
            });
        }

        for (const auto& [dat, target] : marked)
            masks[dat][static_cast<std::size_t>(target)] = 0;
        marked.clear();
    }

    return ncolours;
}

} // namespace op::plan::detail
