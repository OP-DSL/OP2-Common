#include <vector>
#include <limits>
#include <cstdio>

#include "op_seq.h"

constexpr int size = 20;

void min_kernel(const int *d, int *min) {
    *min = std::min(*d, *min);
}

int main(int argc, char **argv) {
    op_init(argc, argv, 2);

    auto s1 = op_decl_set(size, "s1");
    auto s2 = op_decl_set(size / 2, "s2");

    int actual_min = std::numeric_limits<int>::max();
    std::vector<int> data;

    for (int i = 0; i < size; ++i) {
        int v = std::pow(i - (size / 2), 2) + 5;

        data.push_back(v);
        actual_min = std::min(v, actual_min);

        std::printf("i: %d, v: %d\n", i, v);
    }

    auto d1 = op_decl_dat(s1, 1, "int", data.data(), "d1");

    std::vector<int> map_data;
    for (int i = 0; i < size / 2; ++i) {
        map_data.push_back(i * 2);
    }

    auto m1 = op_decl_map(s2, s1, 1, map_data.data(), "m1");

    op_partition("RANDOM", nullptr, nullptr, nullptr, nullptr);

    int min = std::numeric_limits<int>::max();
    op_par_loop(min_kernel, "min_kernel", s2,
            op_arg_dat(d1, 0, m1, 1, "int", OP_READ),
            op_arg_gbl(&min, 1, "int", OP_MIN));

    std::printf("min: %d (expected: %d)\n", min, actual_min);
    op_exit();

    if (min != actual_min) {
        exit(1);
    }
}
