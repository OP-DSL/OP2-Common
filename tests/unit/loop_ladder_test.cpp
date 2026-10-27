// The strategy ladder without a backend: fake strategies answer ready,
// pending or unavailable as scripted, for each of their kernels.

#include <loop/jit.h>
#include <loop/strategy.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../test_support.h"

namespace loop = op::loop;

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
    // The estimates it gives, and the kind its latest prepare was asked for.
    loop::Estimate offline_estimate{true, 1.0f};
    loop::Estimate jit_estimate;
    loop::KernelKind kind = loop::KernelKind::offline;

    explicit Fake(const loop::Point& point)
        : fake_name{point.name}, values{point.values} {}

    std::string_view name() const override { return fake_name; }

    loop::Estimate estimate(loop::KernelKind asked) const override {
        return asked == loop::KernelKind::offline ? offline_estimate : jit_estimate;
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

    std::optional<float> run(const loop::Prepared& prepared,
                             const loop::Launch& launch) override {
        CHECK(prepared.candidate->strategy == this);
        ++runs;
        if (launch.timed)
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
    loop::Ladder ladder{"test_loop"};
    Fake *chunk_atomics = add(ladder, "chunk_atomics", false);
    Fake *atomics = add(ladder, "atomics", true);
    Fake *colouring = add(ladder, "colouring", true);
};

op_set_core source{};
std::array<op_arg, 1> args{};

// An indirect increment: the call runs the exec halo.
loop::Call indirect_call() {
    return {&source, args, source.size + source.exec_size};
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
    CHECK(loop::Call(&source, other, all).key() != key);
    other[0].acc = OP_INC;
    other[0].opt = 0;
    CHECK(loop::Call(&source, other, all).key() != key);
    CHECK(loop::Call(&source, other, all).describe() ==
          "edges (128), inactive args 0");
    op_set_core elsewhere = source;
    elsewhere.index = 7;
    CHECK(loop::Call(&elsewhere, args, all).key() != key);
    CHECK(loop::Call(&source, args, source.size).key() != key);

    // Another dat in its place keeps the key, as a temporary declared again
    // does; which arguments share a dat changes it.
    op_dat_core again = dat;
    again.index = 9;
    other = args;
    other[0].dat = &again;
    CHECK(loop::Call(&source, other, all).key() == key);
    std::array<op_arg, 2> pair{args[0], args[0]};
    pair[1].idx = 1;
    auto shared = loop::Call(&source, pair, all).key();
    pair[1].dat = &again;
    CHECK(loop::Call(&source, pair, all).key() != shared);

    std::array<op_arg, 2> reduction{args[0], op_arg{}};
    reduction[1].opt = 1;
    reduction[1].argtype = OP_ARG_GBL;
    reduction[1].acc = OP_INC;
    loop::Call reduced{&source, reduction, all};
    CHECK(reduced.global_reduction());
    CHECK(reduced.describe() == "edges (128) via edge_nodes, reduction");

    // Sections: a direct loop waits before its one section; an indirect one
    // runs core, then the rest once the halo has arrived, the owned elements
    // on their own when a global reduction follows them.
    auto direct = loop::Sections::direct(&source);
    CHECK(direct.size() == 1 && direct[0].begin == 0 && direct[0].end == 128);
    CHECK(direct.wait_index() == 0 && direct.globals_index() == 0);
    CHECK(direct.set_stride() == 128);

    auto indirect = loop::Sections::indirect(&source, false, all);
    CHECK(indirect.size() == 2);
    CHECK(indirect[0].end == 100 && indirect[1].begin == 100 && indirect[1].end == 136);
    CHECK(indirect.wait_index() == 1 && indirect.globals_index() == 1);
    CHECK(indirect.set_stride() == 160);

    auto separate = loop::Sections::indirect(&source, true, all);
    CHECK(separate.size() == 3);
    CHECK(separate[1].begin == 100 && separate[1].end == 128 && separate[2].end == 136);
    CHECK(separate.wait_index() == 1 && separate.globals_index() == 1);

    // A call's own: the owned elements apart only when it reduces, the exec
    // halo only when the exchange runs it.
    CHECK(loop::Sections::of(one, true).size() == 1);
    CHECK(loop::Sections::of(one, false).size() == 2);
    CHECK(loop::Sections::of(reduced, false).size() == 3);
    auto owned = loop::Sections::of(loop::Call(&source, args, source.size), false);
    CHECK(owned.size() == 2 && owned[1].end == 128);
    auto owned_reduced =
        loop::Sections::of(loop::Call(&source, reduction, source.size), false);
    CHECK(owned_reduced.size() == 3 && owned_reduced[2].size() == 0);

    // Element types in either language's spelling.
    CHECK(loop::scalar_type("double") == loop::ScalarType::f64);
    CHECK(loop::scalar_type("real(8)") == loop::ScalarType::f64);
    CHECK(loop::scalar_type("r4") == loop::ScalarType::f32);
    CHECK(loop::scalar_type("integer(4)") == loop::ScalarType::i32);
    CHECK(loop::scalar_type("logical") == loop::ScalarType::logical);
    CHECK(loop::scalar_type("char") == loop::ScalarType::unknown);
    CHECK(loop::scalar_type(nullptr) == loop::ScalarType::unknown);
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

    loop::Launch launch{args.data(), 1, true};
    CHECK(fixture.atomics->run(fixture.ladder[1].prepare(indirect_call(),
                                                         loop::Urgency::later),
                               launch) == 1.0f);
}

void test_variables() {
    // OP_<FAMILY> enables or turns off a family.
    setenv("OP_CHUNK_ATOMICS", "on", 1);
    setenv("OP_ATOMICS", "off", 1);
    {
        loop::Ladder ladder{"test_loop"};
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
        loop::Ladder ladder{"test_loop"};
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
        loop::Ladder ladder{"test_loop"};
        ladder.add("atomics",
                   [](const loop::Point& point) { return std::make_unique<Fake>(point); });
        CHECK(!ladder[0].enabled && !!ladder[0].off);
    }

    // OP_STRATEGY forces a family for every loop, or for a named loop, even
    // when the family is disabled; its kernel is waited for, and the ladder
    // falls through from it only when it cannot run.
    setenv("OP_STRATEGY", "other:colouring,test_loop:atomics", 1);
    loop::init();
    {
        loop::Ladder ladder{"test_loop"};
        add(ladder, "chunk_atomics", true);
        add(ladder, "atomics", false);
        add(ladder, "plain", true);
        add(ladder, "colouring", true);
        CHECK(ladder.forced() == 1);
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
        loop::Ladder ladder{"other"};
        add(ladder, "atomics", true);
        add(ladder, "colouring", true);
        CHECK(ladder.forced() == 1);
        CHECK(ladder.select(indirect_call()).candidate->name == "colouring=on");
    }
    {
        loop::Ladder ladder{"third"};
        add(ladder, "atomics", true);
        add(ladder, "colouring", true);
        CHECK(!ladder.forced().has_value());
    }

    // A bare family forces every loop that has it; a loop's own item wins.
    setenv("OP_STRATEGY", "atomics,other:colouring", 1);
    loop::init();
    for (const char *name : {"test_loop", "other"}) {
        loop::Ladder ladder{name};
        add(ladder, "atomics", true);
        add(ladder, "colouring", true);
        CHECK(ladder.forced() == (std::string_view{name} == "other" ? 1u : 0u));
    }
    unsetenv("OP_STRATEGY");
    loop::init();
}

// A family's axes: a strategy per point, named by every axis's value, the
// one of first values running and the rest left to the tuner.
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
        loop::Ladder ladder{"test_loop"};
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
        loop::Ladder ladder{"test_loop"};
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
        CHECK(ladder[0].name.starts_with("chunk_atomics="));

        // A tuner's variant is never asked by the ladder.
        static_cast<Fake&>(*ladder[3].strategy).statuses = {loop::Status::unavailable};
        CHECK(ladder.select(indirect_call()).candidate == &ladder[6]);
        for (std::size_t i : {0u, 1u, 2u, 4u, 5u})
            CHECK(static_cast<Fake&>(*ladder[i].strategy).prepares == 0);
    }
    unsetenv("OP_ATOMICS_BLOCK_SIZES");

    // OP_STRATEGY forces a family's running point; a variant's name forces
    // nothing.  OP_<FAMILY>=off turns every point off.
    setenv("OP_STRATEGY", "chunk_atomics,test_loop:colouring=on;block_size=64", 1);
    setenv("OP_COLOURING", "off", 1);
    loop::init();
    declare_all(false);
    {
        loop::Ladder ladder{"test_loop"};
        ladder.add("colouring", make);
        ladder.add("chunk_atomics", make);
        CHECK(ladder.forced() == 3);
        CHECK(ladder.select(indirect_call()).candidate == &ladder[3]);
        CHECK(!!ladder[4].off && !!ladder[5].off);
    }
    unsetenv("OP_STRATEGY");
    unsetenv("OP_COLOURING");
    unsetenv("OP_BLOCK_SIZES");
    unsetenv("OP_CHUNK_ATOMICS_UNSTAGED_SINGLES");
    loop::init();
}

// A ladder longer than any fixed bound: every strategy before the last is
// passed over, disabled or rejected, and the last runs.
void test_long_ladder() {
    loop::Ladder ladder{"long_loop"};
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

// A strategy has a candidate per kernel, the specialised one first when it
// is expected faster or under OP_JIT_FORCE, and an unavailable offline one
// when it has neither; with the JIT off it has no specialised one.
void test_kernel_kinds() {
    using loop::KernelKind;
    using Kinds = std::vector<KernelKind>;
    auto kinds = [](loop::Estimate offline, loop::Estimate jit) {
        declare({{"colouring", {}}});
        loop::Ladder ladder{"test_loop"};
        ladder.add("colouring", [&](const loop::Point& point) {
            auto made = std::make_unique<Fake>(point);
            made->offline_estimate = offline;
            made->jit_estimate = jit;
            return made;
        });
        Kinds out;
        for (std::size_t i = 0; i < ladder.size(); ++i)
            out.push_back(ladder[i].kind);
        return out;
    };
    const auto offline = KernelKind::offline;
    const auto jit = KernelKind::jit;
    const loop::Estimate none{};
    const loop::Estimate unknown{true, std::nullopt};

    loop::init();
    CHECK(kinds({true, 1.0f}, {true, 0.9f}) == (Kinds{jit, offline}));
    CHECK(kinds({true, 1.0f}, {true, 1.0f}) == (Kinds{offline, jit}));
    CHECK(kinds({true, 1.0f}, {true, 1.1f}) == (Kinds{offline, jit}));
    CHECK(kinds({true, 1.0f}, unknown) == (Kinds{offline, jit}));
    CHECK(kinds(unknown, {true, 0.9f}) == (Kinds{offline, jit}));
    CHECK(kinds(none, {true, 1.1f}) == (Kinds{jit}));
    CHECK(kinds({true, 1.0f}, none) == (Kinds{offline}));
    CHECK(kinds(none, none) == (Kinds{offline}));
    setenv("OP_JIT_FORCE", "1", 1);
    loop::init();
    CHECK(kinds({true, 1.0f}, {true, 1.1f}) == (Kinds{jit, offline}));
    CHECK(kinds({true, 1.0f}, none) == (Kinds{offline}));
    setenv("OP_JIT_LEVEL", "0", 1);
    loop::init();
    CHECK(kinds(none, {true, 0.9f}) == (Kinds{offline}));
    unsetenv("OP_JIT_LEVEL");
    unsetenv("OP_JIT_FORCE");
    loop::init();

    // A specialised kernel still building is passed over for the same
    // strategy's offline one, here the floor.
    declare({{"colouring", {}}});
    loop::Ladder ladder{"test_loop"};
    Fake *fake = nullptr;
    ladder.add("colouring", [&](const loop::Point& point) {
        auto made = std::make_unique<Fake>(point);
        made->jit_estimate = {true, 0.9f};
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

} // namespace

int main() {
    try {
        test_call();
        test_defaults();
        test_variables();
        test_axes();
        test_long_ladder();
        test_kernel_kinds();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop ladder tests passed\n");
    return EXIT_SUCCESS;
}
