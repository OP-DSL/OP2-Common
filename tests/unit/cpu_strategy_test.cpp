// The OpenMP strategies against a serial reference, with hand-written
// bodies: every strategy increments a node dat through a two-component map
// and reduces a sum and a maximum, the maximum's element carried as info.
// The edges split into core and owned sections, as a partition's do; the
// MPI functional tests run the exec halo.

#include <cpu/loop.h>
#include <kernel/strategy.h>
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

#include "../test_support.h"

namespace f2c = op::f2c;
namespace cpu = op::cpu;

namespace {

constexpr int edges = 20000;
constexpr int core_edges = 12345;
constexpr int nodes = 3000;

// One element: both targets get the edge's weight and a count, and the
// reductions see the weight.
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
    if (w[n] > *max) {
        *max = w[n];
        *loc = n;
    }
}

template<bool atomic, bool ordered>
void range_body(const cpu::Work& work, op_arg *args) {
    auto *x = reinterpret_cast<double *>(args[0].data);
    const int *map = args[0].map_data;
    auto *w = reinterpret_cast<const double *>(args[2].data);
    for (int i = work.begin; i < work.end; ++i) {
        int n = ordered ? work.order[i] : i;
        element(n, x, map, w, work.global<double>(3), work.global<double>(4),
                work.global<int>(5), [](double& t, double v) {
                    if constexpr (atomic)
                        f2c::atomic_inc(t, v);
                    else
                        f2c::plain_inc(t, v);
                });
    }
}

// A chunk accumulates into its staged slots and flushes them atomically.
void staged_body(const cpu::Work& work, op_arg *args) {
    const auto& staging = work.plan->staging;
    auto *x = reinterpret_cast<double *>(args[0].data);
    const int *map = args[0].map_data;
    auto *w = reinterpret_cast<const double *>(args[2].data);

    char *cursor = work.scratch;
    const int first = staging.slot_offsets[static_cast<std::size_t>(work.chunk)];
    const int count = staging.slot_offsets[static_cast<std::size_t>(work.chunk) + 1] - first;
    double *stage = f2c::stage_region<double>(cursor, count, 2);
    f2c::stage_clear(stage, count * 2, 0, 1);

    const auto stride = static_cast<std::size_t>(staging.stride);
    for (int n = work.begin; n < work.end; ++n)
        element(n, x, map, w, work.global<double>(3), work.global<double>(4),
                work.global<int>(5), [](double& t, double v) { t += v; },
                stage + 2 * staging.stage_words[static_cast<std::size_t>(n)],
                stage + 2 * staging.stage_words[stride + static_cast<std::size_t>(n)]);

    for (int s = 0; s < count; ++s) {
        int target = staging.slot_targets[static_cast<std::size_t>(first + s)];
        f2c::stage_flush(f2c::Ptr<double>{x + 2 * target}, stage, s, 2);
    }
}

constexpr std::array<op::plan::ArgGroup, 2> groups{{{0, 0}, {1, 0}}};

void run(const char *strategy) {
    setenv("OP_STRATEGY", strategy, 1);
    op::loop::init();
    cpu::init();

    op_set_core source{};
    source.index = 0;
    source.size = edges;
    source.core_size = core_edges;
    source.name = "edges";
    op_set_core target{};
    target.index = 1;
    target.size = nodes;
    target.core_size = nodes;

    op_dat_core x_dat{};
    x_dat.index = 0;
    x_dat.set = &target;
    x_dat.dim = 2;
    x_dat.size = 2 * sizeof(double);
    x_dat.type = "double";
    op_dat_core w_dat{};
    w_dat.index = 1;
    w_dat.set = &source;
    w_dat.dim = 1;
    w_dat.size = sizeof(double);
    w_dat.type = "double";

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
    op_map_core map{};
    map.index = 0;
    map.from = &source;
    map.to = &target;
    map.dim = 2;
    map.map = map_values.data();

    std::vector<double> x(2 * nodes, 1.0);
    double sum = 0.5;
    double max = -1.0;
    int loc = -1;

    auto dat_arg = [&](op_dat dat, int idx, op_access access, void *data) {
        op_arg arg{};
        arg.opt = 1;
        arg.argtype = OP_ARG_DAT;
        arg.dat = dat;
        arg.map = idx >= 0 ? &map : nullptr;
        arg.map_data = idx >= 0 ? map_values.data() : nullptr;
        arg.idx = idx;
        arg.dim = dat->dim;
        arg.size = dat->size;
        arg.data = static_cast<char *>(data);
        arg.type = "double";
        arg.acc = access;
        return arg;
    };
    auto gbl_arg = [](void *data, const char *type, int size, op_access access) {
        op_arg arg{};
        arg.opt = 1;
        arg.argtype = OP_ARG_GBL;
        arg.dim = 1;
        arg.size = size;
        arg.data = static_cast<char *>(data);
        arg.type = type;
        arg.acc = access;
        return arg;
    };
    std::array<op_arg, 6> args{
        dat_arg(&x_dat, 0, OP_INC, x.data()),
        dat_arg(&x_dat, 1, OP_INC, x.data()),
        dat_arg(&w_dat, -1, OP_READ, w.data()),
        gbl_arg(&sum, "double", sizeof(double), OP_INC),
        gbl_arg(&max, "double", sizeof(double), OP_MAX),
        gbl_arg(&loc, "int", sizeof(int), OP_READ),
    };
    args[5].argtype = OP_ARG_INFO;
    args[5].acc = 4;

    cpu::Loop loop("test_loop", "openmp_f2c", "Indirect", cpu::LoopDescription{false});
    op::plan::ArgGroups arg_groups{groups, 1};
    loop.add<cpu::ChunkAtomics>("chunk_atomics", op::loop::Default::disabled,
                                &staged_body, arg_groups);
    loop.add<cpu::Simple>("atomics", op::loop::Default::disabled,
                          &range_body<true, false>);
    loop.add<cpu::ChunkColouring>("chunk_colouring", op::loop::Default::enabled,
                                  &range_body<false, false>);
    loop.add<cpu::Colouring>("colouring", op::loop::Default::enabled,
                             &range_body<false, true>);
    // The forced strategy itself runs, not one the ladder falls through to.
    auto forced = loop.ladder().forced();
    CHECK(forced.has_value());
    op::loop::Configuration configuration{&source, args, source.size + source.exec_size};
    CHECK(loop.ladder()[*forced].prepare(configuration, op::loop::Urgency::now,
                                         op::loop::KernelKind::offline).status ==
          op::loop::Status::ready);

    // Twice, so the second call reuses the cached plan.
    std::vector<double> expected(x);
    double expected_sum = sum;
    double expected_max = max;
    int expected_loc = loc;
    for (int call = 0; call < 2; ++call) {
        for (int n = 0; n < edges; ++n)
            element(n, expected.data(), map_values.data(), w.data(), &expected_sum,
                    &expected_max, &expected_loc, [](double& t, double v) { t += v; });
        loop.invoke(&source, args.data(), static_cast<int>(args.size()));
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
