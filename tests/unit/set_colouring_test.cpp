// The set colouring: every colour's elements update distinct targets, the
// owned elements take the first colours when asked, and only the owned
// elements run when nothing updates the halo.

#include <plan/colouring.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "../test_support.h"
#include "builders.h"

using op::test::dat_arg;
using op::test::initialize_dat;
using op::test::initialize_map;
using op::test::initialize_set;

namespace {

// 8 owned elements (4 core) and 2 exec-halo ones over two updated dats.
void test_set_colouring() {
    op_set_core source{};
    op_set_core target_set{};
    initialize_set(source, 8, 4, 2);
    initialize_set(target_set, 5);

    op_dat_core dat_a{};
    op_dat_core dat_b{};
    op_dat_core dat_c{};
    initialize_dat(dat_a, &target_set, 1, sizeof(double), "double");
    initialize_dat(dat_b, &target_set, 2, sizeof(double), "double");
    initialize_dat(dat_c, &target_set, 1, sizeof(double), "double");

    std::vector<int> map_a{0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<int> map_b{1, 2, 3, 4, 0, 1, 2, 3, 4, 0};
    op_map_core a{};
    op_map_core b{};
    initialize_map(a, &source, &target_set, map_a);
    initialize_map(b, &source, &target_set, map_b);

    std::vector<op_arg> args{
        dat_arg(&dat_a, &a, 0, OP_INC),
        dat_arg(&dat_b, &b, 0, OP_RW),
        dat_arg(&dat_c, &a, 0, OP_READ),
    };

    auto check = [&](bool separate_owned, int exec_length) {
        // The maps whose targets a colour's elements must not share.
        std::vector<const std::vector<int> *> updated;
        for (const auto& arg : args)
            if (arg.idx >= 0 && (arg.acc == OP_INC || arg.acc == OP_RW))
                updated.push_back(arg.map == &a ? &map_a : &map_b);

        auto sections = op::loop::Sections::indirect(&source, separate_owned, exec_length);
        auto colouring = op::plan::colour_set(args, sections);
        CHECK(colouring.colours >= 1);
        CHECK(static_cast<int>(colouring.order.size()) == exec_length);
        CHECK(colouring.offsets.size() ==
              static_cast<std::size_t>(colouring.colours) + 1);
        CHECK(colouring.offsets.front() == 0);
        CHECK(colouring.offsets.back() == exec_length);

        std::vector<int> seen(static_cast<std::size_t>(exec_length), 0);
        for (int colour = 0; colour < colouring.colours; ++colour) {
            int begin = colouring.offsets[static_cast<std::size_t>(colour)];
            int end = colouring.offsets[static_cast<std::size_t>(colour) + 1];
            CHECK(begin < end);
            for (int i = begin; i < end; ++i) {
                int element = colouring.order[static_cast<std::size_t>(i)];
                CHECK(element >= 0 && element < exec_length);
                CHECK(seen[static_cast<std::size_t>(element)]++ == 0);
                if (separate_owned && colour < colouring.owned_colours)
                    CHECK(element < source.size);
                // No two elements of a colour update one target.
                for (int j = begin; j < i; ++j) {
                    int other = colouring.order[static_cast<std::size_t>(j)];
                    for (const auto *map : updated)
                        CHECK((*map)[static_cast<std::size_t>(element)] !=
                              (*map)[static_cast<std::size_t>(other)]);
                }
            }
        }
        return colouring;
    };

    auto whole = check(false, 10);
    CHECK(whole.owned_colours == whole.colours);

    auto owned_first = check(true, 10);
    CHECK(owned_first.owned_colours < owned_first.colours);

    // Only reads reach the halo, so the exchange runs the owned elements
    // alone, and only they are coloured.
    for (auto& arg : args)
        arg.acc = OP_READ;
    args[0].acc = OP_INC;
    args[0].map = nullptr;
    args[0].idx = -1;
    args[0].map_data = nullptr;
    auto owned_only = check(false, 8);
    CHECK(owned_only.colours == 1);
}

} // namespace

int main() {
    try {
        test_set_colouring();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("set colouring tests passed\n");
    return EXIT_SUCCESS;
}
