// The dependency graph of a run of calls: edges by kind, levels, launches,
// halo needs, and the next-run predictor.

#include <loop/dag.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace dag = op::loop::dag;
namespace fusion = op::loop::fusion;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

// Airfoil's sets and dats.
struct Mesh {
    op_set_core cells{}, edges{}, bedges{}, nodes{};
    op_dat_core x{}, q{}, qold{}, adt{}, res{};
    op_map_core pcell{}, pecell{}, pbecell{};

    Mesh() {
        int index = 0;
        for (auto *set : {&cells, &edges, &bedges, &nodes})
            set->index = index++;
        index = 0;
        for (auto *dat : {&x, &q, &qold, &adt, &res}) {
            dat->index = index++;
            dat->set = &cells;
            dat->dim = 1;
        }
        x.set = &nodes;
        pcell.from = &cells;
        pcell.to = &nodes;
        pecell.from = &edges;
        pecell.to = &cells;
        pbecell.from = &bedges;
        pbecell.to = &cells;
    }
};

op_arg arg(op_dat dat, op_map map, op_access access) {
    op_arg out{};
    out.opt = 1;
    out.argtype = OP_ARG_DAT;
    out.dat = dat;
    out.map = map;
    out.idx = map != nullptr ? 0 : -1;
    out.acc = access;
    return out;
}

dag::Node node(op_set set, std::vector<op_arg> args, bool atomic = true) {
    dag::Node out;
    out.set = set;
    out.atomic = atomic;
    for (const auto& a : args)
        out.exec |= a.map != nullptr && a.acc != OP_READ;
    out.uses = fusion::uses(args);
    return out;
}

void test_airfoil() {
    Mesh m;
    std::vector<dag::Node> nodes = {
        node(&m.cells, {arg(&m.q, nullptr, OP_READ), arg(&m.qold, nullptr, OP_WRITE)}),
        node(&m.cells, {arg(&m.x, &m.pcell, OP_READ), arg(&m.q, nullptr, OP_READ),
                        arg(&m.adt, nullptr, OP_WRITE)}),
        node(&m.edges, {arg(&m.q, &m.pecell, OP_READ), arg(&m.adt, &m.pecell, OP_READ),
                        arg(&m.res, &m.pecell, OP_INC)}),
        node(&m.bedges, {arg(&m.q, &m.pbecell, OP_READ), arg(&m.adt, &m.pbecell, OP_READ),
                         arg(&m.res, &m.pbecell, OP_INC)}),
        node(&m.cells, {arg(&m.qold, nullptr, OP_READ), arg(&m.q, nullptr, OP_WRITE),
                        arg(&m.res, nullptr, OP_RW), arg(&m.adt, nullptr, OP_READ)}),
    };
    auto graph = dag::build(nodes);
    // save_soln and adt_calc / res_calc and bres_calc / update.
    CHECK(graph.level == (std::vector<int>{0, 0, 1, 1, 2}));
    CHECK(graph.levels == 3 && graph.launches == 3);
    // The increments commute: no edge between res_calc and bres_calc.
    for (const auto& edge : graph.edges)
        CHECK(!(edge.from == 2 && edge.to == 3));
    // Halos: x before the run (adt_calc), q before the run and adt after
    // adt_calc (res_calc); bres_calc needs no new one.
    CHECK(graph.halos.size() == 3);
    CHECK(graph.halos[0].dat == &m.x && graph.halos[0].before_run() && graph.halos[0].reader == 1);
    CHECK(graph.halos[1].dat == &m.q && graph.halos[1].before_run() && graph.halos[1].reader == 2);
    CHECK(graph.halos[2].dat == &m.adt && dag::last_writer(graph.halos[2]) == 1 &&
          graph.halos[2].reader == 2);

    // With res_calc unfusable, it takes a launch of its own.
    nodes[2].fusable = false;
    CHECK(dag::build(nodes).launches == 4);
}

void test_chains() {
    Mesh m;
    // A direct write read directly on the same set: one level.
    std::vector<dag::Node> nodes = {
        node(&m.cells, {arg(&m.q, nullptr, OP_WRITE)}),
        node(&m.cells, {arg(&m.q, nullptr, OP_READ), arg(&m.adt, nullptr, OP_WRITE)}),
    };
    auto graph = dag::build(nodes);
    CHECK(graph.edges.size() == 1 && graph.edges[0].element && graph.levels == 1);

    // Both in the exec halo: the read there needs the write's exchange.
    std::vector<dag::Node> exec = {
        node(&m.cells, {arg(&m.q, nullptr, OP_WRITE), arg(&m.res, &m.pcell, OP_INC)}),
        node(&m.cells, {arg(&m.q, nullptr, OP_READ), arg(&m.adt, &m.pcell, OP_INC)}),
    };
    graph = dag::build(exec);
    CHECK(graph.edges.size() == 1 && !graph.edges[0].element && graph.levels == 2);
    CHECK(graph.halos.size() == 1 && dag::last_writer(graph.halos[0]) == 0 &&
          graph.halos[0].reader == 1);

    // Write after a gather: an edge, a barrier.
    std::vector<dag::Node> war = {
        node(&m.edges, {arg(&m.q, &m.pecell, OP_READ), arg(&m.res, &m.pecell, OP_INC)}),
        node(&m.cells, {arg(&m.q, nullptr, OP_WRITE)}),
    };
    graph = dag::build(war);
    CHECK(graph.edges.size() == 1 && !graph.edges[0].element && graph.level[1] == 1);
}

// Zeroed, then incremented from two sets, then gathered: each increment
// needs the zeroing, not the other increment.
void test_increments() {
    Mesh m;
    std::vector<dag::Node> nodes = {
        node(&m.cells, {arg(&m.res, nullptr, OP_WRITE)}),
        node(&m.edges, {arg(&m.q, &m.pecell, OP_READ), arg(&m.res, &m.pecell, OP_INC)}),
        node(&m.bedges, {arg(&m.q, &m.pbecell, OP_READ), arg(&m.res, &m.pbecell, OP_INC)}),
        node(&m.edges, {arg(&m.res, &m.pecell, OP_READ), arg(&m.adt, &m.pecell, OP_INC)}),
    };
    auto graph = dag::build(nodes);
    CHECK(graph.level == (std::vector<int>{0, 1, 1, 2}));
    auto has = [&](int from, int to) {
        for (const auto& edge : graph.edges)
            if (edge.from == from && edge.to == to)
                return true;
        return false;
    };
    CHECK(has(0, 1) && has(0, 2) && has(1, 3) && has(2, 3) && !has(1, 2));
    // The gathered version is the zeroing's and both increments'.
    CHECK(graph.halos.size() == 2 && graph.halos[1].dat == &m.res &&
          graph.halos[1].writers == (std::vector<int>{0, 1, 2}) && graph.halos[1].reader == 3);
}

void test_predictor() {
    dag::Predictor predictor;
    std::uint64_t next = 0;
    CHECK(!predictor.observe(1));
    CHECK(!predictor.observe(2));
    CHECK(!predictor.observe(1));
    CHECK(predictor.observe(2));
    CHECK(predictor.observe(1));
    CHECK(predictor.predict(1, next) && next == 2);
    CHECK(!predictor.observe(3));
    CHECK(predictor.predict(1, next) && next == 3);
}

} // namespace

int main() {
    try {
        test_airfoil();
        test_chains();
        test_increments();
        test_predictor();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }
    std::printf("dag tests passed\n");
    return EXIT_SUCCESS;
}
