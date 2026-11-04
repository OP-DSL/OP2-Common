// The OpenMP strategies against a serial reference, with hand-written
// bodies: every strategy increments a node dat through a two-component map
// and reduces a sum and a two-component maximum, each component's element
// carried as info.
// The edges split into core and owned sections, as a partition's do; the
// MPI functional tests run the exec halo.

#include <op2/rt/context.h>
#include <op2/rt/cpu/executor.h>
#include <op2/rt/cpu/strategy.h>
#include <op2/rt/kernel/cpu/chunk.h>
#include <op2/rt/kernel/cpu/range.h>
#include <op2/rt/kernel/cpu/threads.h>
#include <op2/rt/kernel/staging.h>
#include <op2/rt/lifecycle.h>
#include <op_lib_c.h>
#include <op_profile.h>

#include <omp.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../loop_parts.h"
#include "../test_support.h"
#include "builders.h"

namespace cpu = op::rt::cpu;
using op::test::dat_arg;
using op::test::initialize_dat;
using op::test::initialize_map;
using op::test::initialize_set;

namespace {

constexpr int edges = 20000;
constexpr int core_edges = 12345;
constexpr int nodes = 3000;

// One element: both targets get the edge's weight and a count, the sum sees
// the weight, and the maximum the weight and its negation, each component
// taking the element as info only when it takes the value.
template<typename Inc>
void element(int n, double *x, const int *map, const double *w, double *sum,
             double *max, int *loc, Inc&& inc, double *slot0 = nullptr,
             double *slot1 = nullptr) {
    double *a = slot0 != nullptr ? slot0 : x + 2 * map[2 * n];
    double *b = slot1 != nullptr ? slot1 : x + 2 * map[2 * n + 1];
    inc(a[0], w[n]);
    inc(a[1], 1.0);
    inc(b[0], 2.0 * w[n]);
    inc(b[1], 1.0);
    *sum += w[n];
    const double value[2] = {w[n], -w[n]};
    for (int d = 0; d < 2; ++d)
        if (value[d] > max[d]) {
            max[d] = value[d];
            loc[d] = n;
        }
}

template<bool atomic>
auto inc() {
    return [](double& t, double v) {
        if constexpr (atomic)
            op::rt::kernel::atomic_inc(t, v);
        else
            op::rt::kernel::plain_inc(t, v);
    };
}

template<bool atomic, bool ordered>
void range_run(const op::rt::abi::cpu::Range& range, op_arg *args) {
    auto *x = reinterpret_cast<double *>(args[0].data);
    const int *map = args[0].map_data;
    auto *w = reinterpret_cast<const double *>(args[2].data);
    op::rt::kernel::cpu::range<ordered>(range, [&](const op::rt::abi::cpu::Thread& thread) {
        return [=](int n) {
            element(n, x, map, w, thread.global<double>(3), thread.global<double>(4),
                    thread.global<int>(5), inc<atomic>());
        };
    });
}

void colouring_run(const op::rt::abi::cpu::Chunk& chunks, op_arg *args) {
    auto *x = reinterpret_cast<double *>(args[0].data);
    const int *map = args[0].map_data;
    auto *w = reinterpret_cast<const double *>(args[2].data);
    op::rt::kernel::cpu::chunks<true>(chunks, [&](const op::rt::abi::cpu::Thread& thread) {
        return [=](int, int begin, int end) {
            for (int n = begin; n < end; ++n)
                element(n, x, map, w, thread.global<double>(3), thread.global<double>(4),
                        thread.global<int>(5), inc<false>());
        };
    });
}

// A chunk accumulates into its staged slots and flushes them atomically.
void staged_run(const op::rt::abi::cpu::Chunk& chunks, op_arg *args) {
    const auto& staging = chunks.staging;
    auto *x = reinterpret_cast<double *>(args[0].data);
    const int *map = args[0].map_data;
    auto *w = reinterpret_cast<const double *>(args[2].data);
    op::rt::kernel::cpu::chunks<false>(chunks, [&](const op::rt::abi::cpu::Thread& thread) {
        return [&, thread](int chunk, int begin, int end) {
            char *cursor = thread.scratch;
            const int first = staging.slot_offsets[static_cast<std::size_t>(chunk)];
            const int count = staging.slot_offsets[static_cast<std::size_t>(chunk) + 1] - first;
            double *stage = op::rt::kernel::stage_region<double>(cursor, count, 2);
            op::rt::kernel::stage_clear(stage, count * 2, 0, 1);

            const auto stride = static_cast<std::size_t>(staging.stride);
            for (int n = begin; n < end; ++n)
                element(n, x, map, w, thread.global<double>(3), thread.global<double>(4),
                        thread.global<int>(5), inc<false>(),
                        stage + 2 * staging.stage_words[static_cast<std::size_t>(n)],
                        stage + 2 * staging.stage_words[stride + static_cast<std::size_t>(n)]);

            for (int s = 0; s < count; ++s) {
                int target = staging.slot_targets[static_cast<std::size_t>(first + s)];
                op::rt::kernel::stage_flush(op::rt::kernel::Ptr<double>{x + 2 * target}, stage, s,
                                            2);
            }
        };
    });
}

constexpr std::array<op::rt::abi::ArgGroup, 2> groups{{{0, 0}, {1, 0}}};

void run(const char *strategy) {
    setenv("OP_STRATEGY", strategy, 1);
    op::rt::init();
    op::rt::init_cpu();

    op_set_core source{};
    op_set_core target{};
    initialize_set(source, edges, core_edges);
    initialize_set(target, nodes);
    source.name = "edges";
    target.index = 1;

    // Neighbouring edges share nodes, as a mesh's do.
    std::mt19937 random(7);
    std::vector<int> map_values(2 * edges);
    std::vector<double> w(edges);
    for (int n = 0; n < edges; ++n) {
        int base = static_cast<int>((static_cast<long long>(n) * nodes) / edges);
        map_values[static_cast<std::size_t>(2 * n)] = base;
        map_values[static_cast<std::size_t>(2 * n + 1)] =
            static_cast<int>((base + 1 + random() % 8) % nodes);
        // Distinct, so the maximum's element does not depend on the order.
        w[static_cast<std::size_t>(n)] =
            static_cast<double>(random() % 1000) / 7.0 + 1e-6 * n;
    }
    // The extremes at either end, so different threads win the two components.
    w.front() = 1000.0;
    w.back() = -1.0;
    op_map_core map{};
    initialize_map(map, &source, &target, map_values, 2);

    std::vector<double> x(2 * nodes, 1.0);
    op_dat_core x_dat{};
    op_dat_core w_dat{};
    initialize_dat(x_dat, &target, 2, sizeof(double), "double");
    initialize_dat(w_dat, &source, 1, sizeof(double), "double");
    w_dat.index = 1;
    x_dat.data = reinterpret_cast<char *>(x.data());
    w_dat.data = reinterpret_cast<char *>(w.data());
    double sum = 0.5;
    std::array<double, 2> max{-1e30, -1e30};
    std::array<int, 2> loc{-1, -1};

    auto gbl_arg = [](void *data, const char *type, int dim, int size, op_access access) {
        op_arg arg{};
        arg.opt = 1;
        arg.argtype = OP_ARG_GBL;
        arg.dim = dim;
        arg.size = dim * size;
        arg.data = static_cast<char *>(data);
        arg.type = type;
        arg.acc = access;
        return arg;
    };
    std::array<op_arg, 6> args{
        dat_arg(&x_dat, &map, 0, OP_INC),
        dat_arg(&x_dat, &map, 1, OP_INC),
        dat_arg(&w_dat, nullptr, -1, OP_READ),
        gbl_arg(&sum, "double", 1, sizeof(double), OP_INC),
        gbl_arg(max.data(), "double", 2, sizeof(double), OP_MAX),
        gbl_arg(loc.data(), "int", 2, sizeof(int), OP_READ),
    };
    args[5].argtype = OP_ARG_INFO;
    args[5].acc = 4;

    const op::rt::abi::cpu::Family families[] = {
        {{"chunk_atomics", op::rt::abi::ArgGroups{groups, 1}}, nullptr, &staged_run},
        {{"atomics", {}}, &range_run<true, false>, nullptr},
        {{"chunk_colouring", {}}, nullptr, &colouring_run},
        {{"colouring", {}}, &range_run<false, true>, nullptr},
    };
    op::rt::abi::cpu::Loop descriptor{
        {.name = "test_loop", .target = "openmp_f2c", .direct = false}, families,
        &op::rt::kernel::cpu::max_threads};
    LoopParts<cpu::Executor> loop{descriptor, op::rt::context()};
    // The forced strategy itself runs, not one the ladder falls through to.
    op::rt::loop::Call call{&source, args, source.size + source.exec_size, false};
    auto selection = loop.ladder.select(call);
    CHECK(selection.status == op::rt::loop::Status::ready &&
          selection.candidate->family == strategy);

    // Twice, so the second call reuses the cached plan.
    std::vector<double> expected(x);
    double expected_sum = sum;
    auto expected_max = max;
    auto expected_loc = loc;
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int n = 0; n < edges; ++n)
            element(n, expected.data(), map_values.data(), w.data(), &expected_sum,
                    expected_max.data(), expected_loc.data(),
                    [](double& t, double v) { t += v; });
        loop.run(call);
    }

    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK(std::abs(x[i] - expected[i]) <= 1e-9 * std::abs(expected[i]));
    CHECK(std::abs(sum - expected_sum) <= 1e-9 * expected_sum);
    CHECK(max == expected_max && loc == expected_loc);
    std::printf("%s passed\n", strategy);
}

} // namespace

int main(int argc, char **argv) {
    op_init(argc, argv, 0);
    op_profile_start("OpenMP strategy test");
    omp_set_num_threads(8);
    for (const char *strategy : {"chunk_atomics", "atomics", "chunk_colouring", "colouring"})
        run(strategy);
    op_profile_end();
    op_exit();
    std::printf("cpu strategy tests passed\n");
    return EXIT_SUCCESS;
}
