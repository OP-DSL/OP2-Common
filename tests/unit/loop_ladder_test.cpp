// The strategy ladder without a backend: fake strategies answer ready,
// pending or unavailable as scripted, for each of their kernels.

#include <op2/rt/lifecycle.h>
#include <op2/rt/loop/ladder.h>
#include <op2/rt/loop/strategy.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../test_support.h"

namespace loop = op::rt::loop;

namespace {

struct Fake : loop::Strategy {
    std::string fake_name;
    std::vector<int> values;
    // Statuses answered in turn, pending becoming ready when waited for;
    // ready once exhausted.  jit_statuses for its specialised kernel.
    std::deque<loop::Status> statuses;
    std::deque<loop::Status> jit_statuses;
    std::string_view reason = "scripted";
    int prepares = 0;
    int waits = 0;
    int runs = 0;
    float time = 1.0f;
    // The estimates it reports, and the kind its latest prepare was asked
    // for.
    loop::Estimate offline_estimate{true, {}, 1.0f};
    loop::Estimate jit_estimate{};
    loop::KernelKind kind = loop::KernelKind::offline;

    explicit Fake(const loop::Point& point)
        : fake_name{point.name}, values{point.values} {}

    std::string_view name() const override { return fake_name; }

    loop::Estimate estimate(loop::KernelKind asked) const override {
        return asked == loop::KernelKind::jit ? jit_estimate : offline_estimate;
    }

    loop::Prepared prepare(const loop::Call&, loop::Urgency urgency,
                           loop::KernelKind asked) override {
        ++prepares;
        kind = asked;
        auto& script = asked == loop::KernelKind::jit ? jit_statuses : statuses;
        loop::Status status = loop::Status::ready;
        if (!script.empty()) {
            status = script.front();
            script.pop_front();
        }
        if (urgency == loop::Urgency::wait) {
            ++waits;
            if (status == loop::Status::pending)
                status = loop::Status::ready;
        }
        loop::Prepared prepared;
        prepared.status = status;
        prepared.reason = status == loop::Status::ready ? "" : reason;
        prepared.entry = this;
        return prepared;
    }

    std::optional<float> run(const loop::Prepared& prepared, const loop::Call&,
                             bool timed) override {
        CHECK(prepared.candidate->strategy == this);
        ++runs;
        if (timed)
            return time;
        return std::nullopt;
    }

    std::string statistics(const loop::Prepared&) const override {
        return "fake";
    }
};

// Declare families, reading their variables now.
void declare(std::initializer_list<loop::Family> families) {
    loop::declare_families({families.begin(), families.size()});
}

// A family without axes, declared now, its one strategy a Fake.
Fake *add(loop::Ladder& ladder, std::string_view family, bool enabled) {
    declare({{family, {}, enabled}});
    Fake *fake = nullptr;
    ladder.add(family, [&](const loop::Point& point) {
        auto made = std::make_unique<Fake>(point);
        fake = made.get();
        return made;
    });
    return fake;
}

// A ladder of chunk_atomics (disabled), atomics, colouring (the floor).
struct Fixture {
    loop::Ladder ladder{"test_loop", false};
    Fake *chunk_atomics = add(ladder, "chunk_atomics", false);
    Fake *atomics = add(ladder, "atomics", true);
    Fake *colouring = add(ladder, "colouring", true);
};

op_set_core source{};
std::array<op_arg, 1> args{};

// An indirect increment: the call runs the exec halo.
loop::Call indirect_call() {
    return {&source, args, source.size + source.exec_size, false};
}

void test_call() {
    op_set_core target{};
    target.index = 1;
    target.size = 64;
    op_dat_core dat{};
    dat.index = 2;
    dat.set = &target;
    op_map_core map{};
    map.index = 3;
    map.name = "edge_nodes";

    source.index = 0;
    source.size = 128;
    source.core_size = 100;
    source.exec_size = 8;
    source.name = "edges";

    args[0].opt = 1;
    args[0].argtype = OP_ARG_DAT;
    args[0].dat = &dat;
    args[0].map = &map;
    args[0].idx = 0;
    args[0].acc = OP_INC;
    args[0].dim = 1;

    auto one = indirect_call();
    CHECK(!one.global_reduction());
    CHECK(one.describe() == "edges (128) via edge_nodes");
    auto key = one.key();
    CHECK(key == indirect_call().key());

    // A different map, access, activity, set or run size changes the key.
    const int all = source.size + source.exec_size;
    std::array<op_arg, 1> other = args;
    other[0].acc = OP_RW;
    CHECK(loop::Call(&source, other, all, false).key() != key);
    other[0].acc = OP_INC;
    other[0].opt = 0;
    CHECK(loop::Call(&source, other, all, false).key() != key);
    CHECK(loop::Call(&source, other, all, false).describe() ==
          "edges (128), inactive args 0");
    op_set_core elsewhere = source;
    elsewhere.index = 7;
    CHECK(loop::Call(&elsewhere, args, all, false).key() != key);
    CHECK(loop::Call(&source, args, source.size, false).key() != key);

    // Another dat in its place keeps the key, as a temporary declared again
    // does; which arguments share a dat changes it.
    op_dat_core again = dat;
    again.index = 9;
    other = args;
    other[0].dat = &again;
    CHECK(loop::Call(&source, other, all, false).key() == key);
    std::array<op_arg, 2> pair{args[0], args[0]};
    pair[1].idx = 1;
    auto shared = loop::Call(&source, pair, all, false).key();
    pair[1].dat = &again;
    CHECK(loop::Call(&source, pair, all, false).key() != shared);

    // A map is listed once, and so is one whose name is part of another's.
    CHECK(loop::Call(&source, pair, all, false).describe() == "edges (128) via edge_nodes");
    op_map_core nodes = map;
    nodes.index = 4;
    nodes.name = "nodes";
    pair[1].map = &nodes;
    CHECK(loop::Call(&source, pair, all, false).describe() ==
          "edges (128) via edge_nodes, nodes");

    std::array<op_arg, 2> reduction{args[0], op_arg{}};
    reduction[1].opt = 1;
    reduction[1].argtype = OP_ARG_GBL;
    reduction[1].acc = OP_INC;
    loop::Call reduced{&source, reduction, all, false};
    CHECK(reduced.global_reduction());
    CHECK(reduced.describe() == "edges (128) via edge_nodes, reduction");

    // Sections: a direct loop waits before its one section; an indirect one
    // runs core, then the rest once the halo has arrived, the owned elements
    // on their own when a global reduction follows them.
    auto direct = loop::Sections::direct(&source);
    CHECK(direct.size() == 1 && direct[0].begin == 0 && direct[0].end == 128);
    CHECK(direct.phases().wait_before == 0 && direct.phases().globals_after == 0);
    CHECK(direct.elements() == 128);

    auto indirect = loop::Sections::indirect(&source, false, all);
    CHECK(indirect.size() == 2);
    CHECK(indirect[0].end == 100 && indirect[1].begin == 100 && indirect[1].end == 136);
    CHECK(indirect.phases().wait_before == 1 && indirect.phases().globals_after == 1);
    CHECK(indirect.elements() == 136);

    auto separate = loop::Sections::indirect(&source, true, all);
    CHECK(separate.size() == 3);
    CHECK(separate[1].begin == 100 && separate[1].end == 128 && separate[2].end == 136);
    CHECK(separate.phases().wait_before == 1 && separate.phases().globals_after == 1);

    // A call's own: the owned elements apart only when it reduces, the exec
    // halo only when the exchange runs it.
    CHECK(loop::Call(&source, args, all, true).sections.size() == 1);
    CHECK(one.sections.size() == 2);
    CHECK(reduced.sections.size() == 3);
    auto owned = loop::Call(&source, args, source.size, false).sections;
    CHECK(owned.size() == 2 && owned[1].end == 128);
    auto owned_reduced = loop::Call(&source, reduction, source.size, false).sections;
    CHECK(owned_reduced.size() == 3 && owned_reduced[2].size() == 0);

    // Element types in either language's spelling.
    CHECK(loop::element_type("double") == op::rt::abi::ElementType::f64);
    CHECK(loop::element_type("real(8)") == op::rt::abi::ElementType::f64);
    CHECK(loop::element_type("r4") == op::rt::abi::ElementType::f32);
    CHECK(loop::element_type("integer(4)") == op::rt::abi::ElementType::i32);
    CHECK(loop::element_type("logical") == op::rt::abi::ElementType::logical);
    CHECK(!loop::element_type("char"));
    CHECK(!loop::element_type(nullptr));
}

void test_defaults() {
    Fixture fixture;
    CHECK(fixture.ladder.size() == 3);
    CHECK(!fixture.ladder[0].enabled && !fixture.ladder[0].off);
    CHECK(fixture.ladder[1].enabled && fixture.ladder[2].enabled);
    CHECK(fixture.ladder[0].name == "chunk_atomics=on" &&
          fixture.ladder[0].kind == loop::KernelKind::offline);

    // The first enabled strategy that is ready runs; a disabled one is
    // never asked.
    auto selection = fixture.ladder.select(indirect_call());
    CHECK(selection.candidate->strategy == fixture.atomics);
    CHECK(selection.status == loop::Status::ready);
    CHECK(fixture.chunk_atomics->prepares == 0);
    CHECK(fixture.colouring->prepares == 0);

    // Pending or rejected, the ladder moves on.
    fixture.atomics->statuses = {loop::Status::pending, loop::Status::unavailable};
    selection = fixture.ladder.select(indirect_call());
    CHECK(selection.candidate->strategy == fixture.colouring);
    selection = fixture.ladder.select(indirect_call());
    CHECK(selection.candidate->strategy == fixture.colouring);
    selection = fixture.ladder.select(indirect_call());
    CHECK(selection.candidate->strategy == fixture.atomics);

    // With nothing ready the floor is waited for, so a selection is always
    // ready.
    fixture.atomics->statuses = {loop::Status::unavailable};
    fixture.colouring->statuses = {loop::Status::pending, loop::Status::pending};
    int before = fixture.colouring->prepares;
    selection = fixture.ladder.select(indirect_call());
    CHECK(selection.candidate->strategy == fixture.colouring);
    CHECK(selection.status == loop::Status::ready);
    CHECK(fixture.colouring->prepares == before + 2 && fixture.colouring->waits == 1);

    auto call = indirect_call();
    CHECK(fixture.atomics->run(fixture.ladder[1].prepare(call, loop::Urgency::later), call,
                               true) == 1.0f);
}

void test_variables() {
    // OP_<FAMILY> enables or turns off a family.
    setenv("OP_CHUNK_ATOMICS", "on", 1);
    setenv("OP_ATOMICS", "off", 1);
    {
        loop::Ladder ladder{"test_loop", false};
        add(ladder, "chunk_atomics", false);
        add(ladder, "atomics", true);
        add(ladder, "colouring", true);
        CHECK(ladder[0].enabled);
        CHECK(!ladder[1].enabled && !!ladder[1].off);
        CHECK(ladder[2].enabled);
        CHECK(ladder.select(indirect_call()).candidate->name == "chunk_atomics=on");
    }
    unsetenv("OP_CHUNK_ATOMICS");
    unsetenv("OP_ATOMICS");

    // An unusable value is ignored, leaving the default.
    setenv("OP_CHUNK_ATOMICS", "maybe", 1);
    setenv("OP_ATOMICS", "maybe", 1);
    {
        loop::Ladder ladder{"test_loop", false};
        add(ladder, "chunk_atomics", false);
        add(ladder, "atomics", true);
        CHECK(!ladder[0].enabled && !ladder[0].off);
        CHECK(ladder[1].enabled && !ladder[1].off);
    }
    unsetenv("OP_CHUNK_ATOMICS");
    unsetenv("OP_ATOMICS");

    // A family's variables are read when it is declared, not when a loop
    // registers it.
    setenv("OP_ATOMICS", "off", 1);
    declare({{"atomics", {}}});
    unsetenv("OP_ATOMICS");
    {
        loop::Ladder ladder{"test_loop", false};
        ladder.add("atomics",
                   [](const loop::Point& point) { return std::make_unique<Fake>(point); });
        CHECK(!ladder[0].enabled && !!ladder[0].off);
    }

    // OP_STRATEGY forces a family for every loop, or for a named loop, even
    // when the family is disabled; its kernel is waited for, and the ladder
    // falls through from it only when it cannot run.
    setenv("OP_STRATEGY", "other:colouring,test_loop:atomics", 1);
    op::rt::init();
    {
        loop::Ladder ladder{"test_loop", false};
        add(ladder, "chunk_atomics", true);
        add(ladder, "atomics", false);
        add(ladder, "plain", true);
        add(ladder, "colouring", true);
        auto selection = ladder.select(indirect_call());
        CHECK(selection.candidate->name == "atomics=on");

        auto& atomics = static_cast<Fake&>(*ladder[1].strategy);
        atomics.statuses = {loop::Status::pending};
        selection = ladder.select(indirect_call());
        CHECK(selection.candidate->name == "atomics=on");
        CHECK(atomics.waits == atomics.prepares);

        atomics.statuses = {loop::Status::unavailable};
        selection = ladder.select(indirect_call());
        CHECK(selection.candidate->name == "plain=on");
        CHECK(static_cast<Fake&>(*ladder[0].strategy).prepares == 0);
    }
    {
        loop::Ladder ladder{"other", false};
        add(ladder, "atomics", true);
        add(ladder, "colouring", true);
        CHECK(ladder.select(indirect_call()).candidate->name == "colouring=on");
    }
    {
        loop::Ladder ladder{"third", false};
        add(ladder, "atomics", true);
        add(ladder, "colouring", true);
        // Not forced: its first member runs, not waited for.
        CHECK(ladder.select(indirect_call()).candidate->name == "atomics=on");
        CHECK(static_cast<Fake&>(*ladder[0].strategy).waits == 0);
    }

    // A bare family forces every loop that has it; a loop's own item wins.
    setenv("OP_STRATEGY", "atomics,other:colouring", 1);
    op::rt::init();
    for (const char *name : {"test_loop", "other"}) {
        loop::Ladder ladder{name, false};
        add(ladder, "atomics", true);
        add(ladder, "colouring", true);
        // The forced family runs, its kernel waited for.
        auto selection = ladder.select(indirect_call());
        bool other = std::string_view{name} == "other";
        CHECK(selection.candidate->name == (other ? "colouring=on" : "atomics=on"));
        CHECK(static_cast<Fake&>(*selection.candidate->strategy).waits == 1);
    }
    unsetenv("OP_STRATEGY");
    op::rt::init();
}

// A family's axes: a strategy per point, named by every axis's value, the
// one of first values running and, in a ladder that explores, the rest left to
// the tuner.
void test_axes() {
    const std::vector<loop::Axis> axes{
        {"BLOCK_SIZES", "block_size", false, {128}},
        {"UNSTAGED_SINGLES", "singles", true, {1}},
    };
    auto make = [](const loop::Point& point) { return std::make_unique<Fake>(point); };
    auto declare_all = [&](bool chunk_atomics) {
        declare({{"chunk_atomics", axes, chunk_atomics},
                 {"atomics", {axes[0]}, false},
                 {"colouring", {axes[0]}}});
    };

    declare_all(true);
    {
        loop::Ladder ladder{"test_loop", false};
        ladder.add("chunk_atomics", make);
        CHECK(ladder.size() == 1 &&
              ladder[0].name == "chunk_atomics=on;block_size=128;singles=on");
        CHECK((static_cast<Fake&>(*ladder[0].strategy).values == std::vector<int>{128, 1}));
    }

    // Lists, without repeats or unusable items; the family's own variable
    // over the axis-wide one.
    setenv("OP_BLOCK_SIZES", "128,64,128,x", 1);
    setenv("OP_CHUNK_ATOMICS_UNSTAGED_SINGLES", "on,off", 1);
    setenv("OP_ATOMICS_BLOCK_SIZES", "256", 1);
    declare_all(true);
    {
        // In declared order, whatever order the loop adds them in.
        loop::Ladder ladder{"test_loop", true};
        ladder.add("colouring", make);
        ladder.add("atomics", make);
        ladder.add("chunk_atomics", make);
        std::vector<std::string> names;
        for (std::size_t i = 0; i < ladder.size(); ++i)
            names.emplace_back(ladder[i].name);
        CHECK((names == std::vector<std::string>{
                            "chunk_atomics=on;block_size=64;singles=on",
                            "chunk_atomics=on;block_size=128;singles=off",
                            "chunk_atomics=on;block_size=64;singles=off",
                            "chunk_atomics=on;block_size=128;singles=on",
                            "atomics=on;block_size=256",
                            "colouring=on;block_size=64",
                            "colouring=on;block_size=128"}));
        for (std::size_t i : {0u, 1u, 2u, 4u, 5u})
            CHECK(!ladder[i].enabled && !ladder[i].off);
        CHECK(ladder[3].enabled && ladder[6].enabled);

        // Summaries label a candidate by its family and the axis values where
        // its point differs from the one the ladder runs.
        CHECK(ladder[0].label == "chunk_atomics (block_size=64)");
        CHECK(ladder[1].label == "chunk_atomics (singles=off)");
        CHECK(ladder[2].label == "chunk_atomics (block_size=64 singles=off)");
        CHECK(ladder[3].label == "chunk_atomics" && ladder[5].label == "colouring (block_size=64)");
        CHECK(ladder[0].name.starts_with("chunk_atomics="));

        // A tuner's variant is never asked by the ladder.
        static_cast<Fake&>(*ladder[3].strategy).statuses = {loop::Status::unavailable};
        CHECK(ladder.select(indirect_call()).candidate == &ladder[6]);
        for (std::size_t i : {0u, 1u, 2u, 4u, 5u})
            CHECK(static_cast<Fake&>(*ladder[i].strategy).prepares == 0);
    }
    {
        // A ladder that does not explore has the points of first values alone.
        loop::Ladder ladder{"test_loop", false};
        ladder.add("colouring", make);
        ladder.add("atomics", make);
        ladder.add("chunk_atomics", make);
        CHECK(ladder.strategies().size() == 3);
        CHECK(ladder.size() == 3 &&
              ladder[0].name == "chunk_atomics=on;block_size=128;singles=on" &&
              ladder[1].name == "atomics=on;block_size=256" &&
              ladder[2].name == "colouring=on;block_size=128");
    }
    unsetenv("OP_ATOMICS_BLOCK_SIZES");

    // OP_STRATEGY forces a family's running point; a variant's name forces
    // nothing.  OP_<FAMILY>=off turns every point off.
    setenv("OP_STRATEGY", "chunk_atomics,test_loop:colouring=on;block_size=64", 1);
    setenv("OP_COLOURING", "off", 1);
    op::rt::init();
    declare_all(false);
    {
        loop::Ladder ladder{"test_loop", true};
        ladder.add("colouring", make);
        ladder.add("chunk_atomics", make);
        CHECK(ladder.select(indirect_call()).candidate == &ladder[3]);
        CHECK(!!ladder[4].off && !!ladder[5].off);
    }
    unsetenv("OP_STRATEGY");
    unsetenv("OP_COLOURING");
    unsetenv("OP_BLOCK_SIZES");
    unsetenv("OP_CHUNK_ATOMICS_UNSTAGED_SINGLES");
    op::rt::init();
}

// A ladder longer than any fixed bound: every strategy before the last is
// passed over, disabled or rejected, and the last runs.
void test_long_ladder() {
    loop::Ladder ladder{"long_loop", false};
    std::vector<Fake *> fakes;
    for (int i = 0; i < 40; ++i) {
        auto *fake = add(ladder, "fake_" + std::to_string(i), i % 2 == 0);
        fake->statuses.assign(4, loop::Status::unavailable);
        fakes.push_back(fake);
    }
    fakes.back()->statuses.clear();
    for (int repeat = 0; repeat < 2; ++repeat) {
        auto selection = ladder.select(indirect_call());
        CHECK(selection.candidate->strategy == fakes.back());
        CHECK(selection.status == loop::Status::ready);
    }
    CHECK(fakes[0]->prepares == 2 && fakes[1]->prepares == 0);
}

// A strategy has a candidate per kernel, the specialised one first when its
// backend estimates it cheaper or under OP_JIT_FORCE, and an unavailable
// offline one when it has neither; with the JIT off it has no specialised
// one.
void test_kernel_kinds() {
    using loop::KernelKind;
    using Kinds = std::vector<KernelKind>;
    // An offline kernel of cost offline, or none when negative, and a
    // specialised one of cost jit, or none when negative; a cost of 0 is
    // none known.
    auto kinds = [](float offline_cost, float jit_cost) {
        auto estimate = [](float cost) {
            if (cost < 0)
                return loop::Estimate{false, "kernel_unavailable", {}};
            return loop::Estimate{true, {}, cost > 0 ? std::optional<float>{cost}
                                                     : std::optional<float>{}};
        };
        declare({{"colouring", {}}});
        loop::Ladder ladder{"test_loop", false};
        ladder.add("colouring", [&](const loop::Point& point) {
            auto made = std::make_unique<Fake>(point);
            made->offline_estimate = estimate(offline_cost);
            made->jit_estimate = estimate(jit_cost);
            return made;
        });
        Kinds out;
        for (std::size_t i = 0; i < ladder.size(); ++i)
            out.push_back(ladder[i].kind);
        return out;
    };
    const auto offline = KernelKind::offline;
    const auto jit = KernelKind::jit;

    op::rt::init();
    CHECK(kinds(1.0f, 0.9f) == (Kinds{jit, offline}));
    CHECK(kinds(1.0f, 1.0f) == (Kinds{offline, jit}));
    CHECK(kinds(1.0f, 1.1f) == (Kinds{offline, jit}));
    CHECK(kinds(0, 0) == (Kinds{offline, jit}));
    CHECK(kinds(-1, 1.0f) == (Kinds{jit}));
    CHECK(kinds(1.0f, -1) == (Kinds{offline}));
    CHECK(kinds(-1, -1) == (Kinds{offline}));
    setenv("OP_JIT_FORCE", "1", 1);
    op::rt::init();
    CHECK(kinds(1.0f, 1.1f) == (Kinds{jit, offline}));
    CHECK(kinds(1.0f, -1) == (Kinds{offline}));
    setenv("OP_JIT_LEVEL", "0", 1);
    op::rt::init();
    CHECK(kinds(-1, 1.0f) == (Kinds{offline}));
    unsetenv("OP_JIT_LEVEL");
    unsetenv("OP_JIT_FORCE");
    op::rt::init();

    // A specialised kernel still building is passed over for the same
    // strategy's offline one, here the floor.
    declare({{"colouring", {}}});
    loop::Ladder ladder{"test_loop", false};
    Fake *fake = nullptr;
    ladder.add("colouring", [&](const loop::Point& point) {
        auto made = std::make_unique<Fake>(point);
        made->jit_estimate = {true, {}, 0.9f};
        fake = made.get();
        return made;
    });
    CHECK(ladder.size() == 2 && ladder[0].name == "colouring=on;kernel=jit");
    fake->jit_statuses = {loop::Status::pending};
    auto selection = ladder.select(indirect_call());
    CHECK(selection.candidate == &ladder[1] && fake->kind == offline);
    selection = ladder.select(indirect_call());
    CHECK(selection.candidate == &ladder[0] && fake->kind == jit);
}

// A point with no kernel of either kind has no candidate, so nothing
// prepares, times or reports it; the ladder's point of the family keeps an
// offline candidate, refused with the estimates' reason, its strategy never
// asked.
void test_kernel_less_points() {
    const loop::Axis axis{"SWITCHES", "switch", true, {1}};
    auto build = [&](loop::Ladder& ladder, std::vector<Fake *>& fakes) {
        declare({{"colouring", {axis}}});
        ladder.add("colouring", [&](const loop::Point& point) {
            auto made = std::make_unique<Fake>(point);
            if (point.value("switch") == 0) {
                made->offline_estimate = {false, "kernel_unavailable", {}};
                made->jit_estimate = {false, "build_failed", {}};
            }
            fakes.push_back(made.get());
            return made;
        });
    };

    // The tuner's point is the one without a kernel: it has no candidate.
    setenv("OP_COLOURING_SWITCHES", "on,off", 1);
    op::rt::init();
    {
        loop::Ladder ladder{"test_loop", true};
        std::vector<Fake *> fakes;
        build(ladder, fakes);
        CHECK(fakes.size() == 2 && ladder.size() == 1);
        CHECK(ladder[0].member && ladder[0].refused.empty());
    }

    // The ladder's point is: it keeps a candidate, refused with the
    // specialised kernel's reason, and its strategy is never asked.
    setenv("OP_COLOURING_SWITCHES", "off,on", 1);
    op::rt::init();
    {
        loop::Ladder ladder{"test_loop", true};
        std::vector<Fake *> fakes;
        build(ladder, fakes);
        CHECK(ladder.size() == 2 && !ladder[0].member && ladder[1].member);
        CHECK(ladder[1].kind == loop::KernelKind::offline &&
              ladder[1].refused == "build_failed");
        auto prepared = ladder[1].prepare(indirect_call(), loop::Urgency::later);
        CHECK(prepared.status == loop::Status::unavailable &&
              prepared.reason == "build_failed" && prepared.candidate == &ladder[1]);
        CHECK(fakes[1]->prepares == 0);
    }
    unsetenv("OP_COLOURING_SWITCHES");
    op::rt::init();
}

} // namespace

int main() {
    try {
        test_call();
        test_defaults();
        test_variables();
        test_axes();
        test_long_ladder();
        test_kernel_kinds();
        test_kernel_less_points();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop ladder tests passed\n");
    return EXIT_SUCCESS;
}
