// The plans a loop's strategies share: keys over what a planner plans from,
// one plan per key while any strategy holds it, and its release once every
// strategy has forgotten the configuration.

#include <plan/cache.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../test_support.h"
#include "builders.h"

namespace plan = op::plan;
namespace loop = op::loop;
using op::test::dat_arg;
using op::test::initialize_dat;
using op::test::initialize_map;
using op::test::initialize_set;

namespace {

// A plan that counts the ones alive.
struct Counted {
    static int alive;
    int id;

    explicit Counted(int id_) : id{id_} { ++alive; }
    Counted(const Counted&) = delete;
    ~Counted() { --alive; }
};

int Counted::alive = 0;

op_set_core edges{};
op_set_core nodes{};
op_dat_core x{};
op_map_core edge_nodes{};
std::vector<int> map_values(8, 0);

loop::Call call_of(std::array<op_arg, 1>& args) {
    return {&edges, args, edges.size};
}

// The planner's name, the call's shape, each argument's bytes and the
// planner's added inputs each make a different key.
void test_keys() {
    std::array<op_arg, 1> args{dat_arg(&x, &edge_nodes, 0, OP_INC)};
    auto call = call_of(args);
    auto key = plan::PlanKey("chunk_atomics", call).value();
    CHECK(plan::PlanKey("chunk_atomics", call).value() == key);
    CHECK(plan::PlanKey("chunk_colouring", call).value() != key);
    CHECK(plan::PlanKey("chunk_atomics", call).add(128).value() != key);
    CHECK(plan::PlanKey("chunk_atomics", call).add(128).value() !=
          plan::PlanKey("chunk_atomics", call).add(256).value());
    CHECK(plan::PlanKey("chunk_atomics", call).add(1.5).value() !=
          plan::PlanKey("chunk_atomics", call).add(2.0).value());

    // A float dat of the same dimension: the call's shape is the same, the
    // plan's staging is not.
    std::array<op_arg, 1> floats = args;
    floats[0].size = static_cast<int>(sizeof(float));
    auto float_call = call_of(floats);
    CHECK(float_call.key() == call.key());
    CHECK(plan::PlanKey("chunk_atomics", float_call).value() != key);

    // An inactive argument's bytes count for nothing.
    std::array<op_arg, 1> inactive = args;
    inactive[0].opt = 0;
    std::array<op_arg, 1> inactive_floats = floats;
    inactive_floats[0].opt = 0;
    CHECK(plan::PlanKey("chunk_atomics", call_of(inactive)).value() ==
          plan::PlanKey("chunk_atomics", call_of(inactive_floats)).value());
}

// Two strategies with plans alike hold one, made once; it goes once both
// forget the configuration, and a strategy that needs it again plans anew.
void test_sharing() {
    std::array<op_arg, 1> args{dat_arg(&x, &edge_nodes, 0, OP_INC)};
    auto call = call_of(args);
    plan::SharedPlans<Counted> shared;
    plan::PlanSlots<Counted> first;
    plan::PlanSlots<Counted> second;
    int made = 0;
    auto get = [&](plan::PlanSlots<Counted>& slots, int block_size) -> const Counted& {
        return slots.get(call, [&] {
            plan::PlanKey key{"chunk_atomics", call};
            key.add(block_size);
            return shared.get(key, [&] { return std::make_shared<Counted>(++made); });
        });
    };

    const auto& a = get(first, 128);
    const auto& b = get(second, 128);
    CHECK(&a == &b && made == 1 && Counted::alive == 1);
    CHECK(&get(first, 128) == &a && made == 1);
    CHECK(shared.live() == 1 && first.size() == 1);

    // A strategy of another block size: a plan of its own.
    plan::PlanSlots<Counted> wide;
    const auto& c = get(wide, 256);
    CHECK(&c != &a && made == 2 && Counted::alive == 2);

    first.forget(call.key());
    CHECK(Counted::alive == 2 && first.size() == 0);
    second.forget(call.key());
    CHECK(Counted::alive == 1 && shared.live() == 1);
    CHECK(get(first, 128).id == 3 && Counted::alive == 2);

    first.clear();
    wide.clear();
    CHECK(Counted::alive == 0 && shared.live() == 0);
}

} // namespace

int main() {
    initialize_set(edges, 4);
    initialize_set(nodes, 5);
    edges.index = 0;
    nodes.index = 1;
    initialize_dat(x, &nodes, 2, sizeof(double), "double");
    initialize_map(edge_nodes, &edges, &nodes, map_values, 2);

    test_keys();
    test_sharing();

    std::printf("plan cache tests passed\n");
    return EXIT_SUCCESS;
}
