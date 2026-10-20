// The strategy ladder without a backend: fake strategies answer ready,
// pending, unavailable or rejected as scripted.

#include <loop/strategy.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace loop = op::loop;

namespace {

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition))                                                       \
            throw std::runtime_error(std::string("check failed: ") +          \
                                     #condition + " at line " +               \
                                     std::to_string(__LINE__));                 \
    } while (false)

struct Fake : loop::Strategy {
    std::string fake_name;
    std::vector<int> values;
    // Statuses answered in turn; ready once exhausted.
    std::deque<loop::Status> statuses;
    std::string_view reason = "scripted";
    int prepares = 0;
    int now_prepares = 0;
    int runs = 0;
    float time = 1.0f;

    explicit Fake(std::string name) : fake_name{std::move(name)} {}
    explicit Fake(const loop::Point& point)
        : fake_name{point.name}, values{point.values} {}

    std::string_view name() const override { return fake_name; }

    loop::Prepared prepare(const loop::Configuration&, bool now) override {
        ++prepares;
        now_prepares += now;
        loop::Status status = loop::Status::ready;
        if (!statuses.empty()) {
            status = statuses.front();
            statuses.pop_front();
        }
        loop::Prepared prepared;
        prepared.strategy = this;
        prepared.status = status;
        prepared.reason = status == loop::Status::ready ? "" : reason;
        prepared.entry = this;
        return prepared;
    }

    std::optional<float> run(const loop::Prepared& prepared,
                             const loop::Launch& launch) override {
        CHECK(prepared.strategy == this);
        ++runs;
        if (launch.timed)
            return time;
        return std::nullopt;
    }

    std::string statistics(const loop::Prepared&) const override {
        return "fake";
    }
};

// A family without axes, its one strategy a Fake.
Fake *add(loop::Ladder& ladder, std::string_view family, loop::Default setting) {
    Fake *fake = nullptr;
    ladder.add(family, setting, {}, [&](const loop::Point& point) {
        auto made = std::make_unique<Fake>(point);
        fake = made.get();
        return made;
    });
    return fake;
}

// A ladder of chunk_atomics (disabled), atomics, colouring (the floor).
struct Fixture {
    loop::Ladder ladder{"test_loop"};
    Fake *chunk_atomics = add(ladder, "chunk_atomics", loop::Default::disabled);
    Fake *atomics = add(ladder, "atomics", loop::Default::enabled);
    Fake *colouring = add(ladder, "colouring", loop::Default::enabled);
};

op_set_core source{};
std::array<op_arg, 1> args{};

loop::Configuration configuration() {
    return {&source, args};
}

void test_configuration() {
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

    auto one = configuration();
    CHECK(!one.global_reduction());
    CHECK(one.describe() == "edges (128) via edge_nodes");
    auto key = one.key();
    CHECK(key == configuration().key());

    // A different map, access, activity or set changes the key.
    std::array<op_arg, 1> other = args;
    other[0].acc = OP_RW;
    CHECK(loop::Configuration(&source, other).key() != key);
    other[0].acc = OP_INC;
    other[0].opt = 0;
    CHECK(loop::Configuration(&source, other).key() != key);
    CHECK(loop::Configuration(&source, other).describe() ==
          "edges (128), inactive args 0");
    op_set_core elsewhere = source;
    elsewhere.index = 7;
    CHECK(loop::Configuration(&elsewhere, args).key() != key);

    std::array<op_arg, 2> reduction{args[0], op_arg{}};
    reduction[1].opt = 1;
    reduction[1].argtype = OP_ARG_GBL;
    reduction[1].acc = OP_INC;
    loop::Configuration reduced{&source, reduction};
    CHECK(reduced.global_reduction());
    CHECK(reduced.describe() == "edges (128) via edge_nodes, reduction");

    // Sections: a direct loop waits before its one section; an indirect one
    // runs core, then the rest once the halo has arrived, the owned elements
    // on their own when a global reduction follows them.
    auto direct = loop::Sections::direct(&source);
    CHECK(direct.size() == 1 && direct[0].begin == 0 && direct[0].end == 128);
    CHECK(direct.wait_index() == 0 && direct.globals_index() == 0);
    CHECK(direct.set_stride() == 128);

    auto indirect = loop::Sections::indirect(&source, false);
    CHECK(indirect.size() == 2);
    CHECK(indirect[0].end == 100 && indirect[1].begin == 100 && indirect[1].end == 136);
    CHECK(indirect.wait_index() == 1 && indirect.globals_index() == 1);
    CHECK(indirect.set_stride() == 160);

    auto separate = loop::Sections::indirect(&source, true);
    CHECK(separate.size() == 3);
    CHECK(separate[1].begin == 100 && separate[1].end == 128 && separate[2].end == 136);
    CHECK(separate.wait_index() == 1 && separate.globals_index() == 1);

    // A call's own: the owned elements apart only when it reduces.
    CHECK(loop::Sections::of(one, true).size() == 1);
    CHECK(loop::Sections::of(one, false).size() == 2);
    CHECK(loop::Sections::of(reduced, false).size() == 3);

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
    CHECK(!fixture.ladder.enabled(0) && fixture.ladder.explorable(0));
    CHECK(fixture.ladder.enabled(1) && fixture.ladder.enabled(2));
    CHECK(fixture.chunk_atomics->family() == "chunk_atomics");

    // The first enabled strategy that is ready runs; a disabled one is
    // never asked.
    auto selection = fixture.ladder.select(configuration());
    CHECK(selection.strategy == fixture.atomics);
    CHECK(selection.prepared.status == loop::Status::ready);
    CHECK(selection.prepared.strategy == fixture.atomics);
    CHECK(fixture.chunk_atomics->prepares == 0);
    CHECK(fixture.colouring->prepares == 0);

    // Pending or rejected, the ladder moves on.
    fixture.atomics->statuses = {loop::Status::pending, loop::Status::rejected};
    selection = fixture.ladder.select(configuration());
    CHECK(selection.strategy == fixture.colouring);
    selection = fixture.ladder.select(configuration());
    CHECK(selection.strategy == fixture.colouring);
    selection = fixture.ladder.select(configuration());
    CHECK(selection.strategy == fixture.atomics);

    // With nothing ready the floor is selected whatever its status, and
    // prepared only once.
    fixture.atomics->statuses = {loop::Status::unavailable};
    fixture.colouring->statuses = {loop::Status::pending};
    int before = fixture.colouring->prepares;
    selection = fixture.ladder.select(configuration());
    CHECK(selection.strategy == fixture.colouring);
    CHECK(selection.prepared.status == loop::Status::pending);
    CHECK(fixture.colouring->prepares == before + 1);

    loop::Launch launch{args.data(), 1, true};
    CHECK(fixture.atomics->run(selection.prepared.strategy == fixture.atomics
                                   ? selection.prepared
                                   : fixture.atomics->prepare(configuration(), false),
                               launch) == 1.0f);
}

void test_variables() {
    // OP_<FAMILY> enables or turns off a family.
    setenv("OP_CHUNK_ATOMICS", "on", 1);
    setenv("OP_ATOMICS", "off", 1);
    {
        loop::Ladder ladder{"test_loop"};
        add(ladder, "chunk_atomics", loop::Default::disabled);
        add(ladder, "atomics", loop::Default::enabled);
        add(ladder, "colouring", loop::Default::enabled);
        CHECK(ladder.enabled(0));
        CHECK(!ladder.enabled(1) && !ladder.explorable(1));
        CHECK(ladder.enabled(2));
        CHECK(ladder.select(configuration()).strategy->name() == "chunk_atomics=on");
    }
    unsetenv("OP_CHUNK_ATOMICS");
    unsetenv("OP_ATOMICS");

    // OP_STRATEGY forces a family for every loop, or for a named loop, even
    // when the family is disabled; the ladder falls through from it.
    setenv("OP_STRATEGY", "other:colouring,test_loop:atomics", 1);
    loop::init();
    {
        loop::Ladder ladder{"test_loop"};
        add(ladder, "chunk_atomics", loop::Default::enabled);
        add(ladder, "atomics", loop::Default::disabled);
        add(ladder, "plain", loop::Default::enabled);
        add(ladder, "colouring", loop::Default::enabled);
        CHECK(ladder.forced() == 1);
        auto selection = ladder.select(configuration());
        CHECK(selection.strategy->name() == "atomics=on");

        static_cast<Fake&>(ladder[1]).statuses = {loop::Status::pending};
        selection = ladder.select(configuration());
        CHECK(selection.strategy->name() == "plain=on");
        CHECK(static_cast<Fake&>(ladder[0]).prepares == 0);
    }
    {
        loop::Ladder ladder{"other"};
        add(ladder, "atomics", loop::Default::enabled);
        add(ladder, "colouring", loop::Default::enabled);
        CHECK(ladder.forced() == 1);
        CHECK(ladder.select(configuration()).strategy->name() == "colouring=on");
    }
    {
        loop::Ladder ladder{"third"};
        add(ladder, "atomics", loop::Default::enabled);
        add(ladder, "colouring", loop::Default::enabled);
        CHECK(!ladder.forced().has_value());
    }

    // A bare family forces every loop that has it; a loop's own item wins.
    setenv("OP_STRATEGY", "atomics,other:colouring", 1);
    loop::init();
    for (const char *name : {"test_loop", "other"}) {
        loop::Ladder ladder{name};
        add(ladder, "atomics", loop::Default::enabled);
        add(ladder, "colouring", loop::Default::enabled);
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

    {
        loop::Ladder ladder{"test_loop"};
        ladder.add("chunk_atomics", loop::Default::enabled, axes, make);
        CHECK(ladder.size() == 1 &&
              ladder[0].name() == "chunk_atomics=on;block_size=128;singles=on");
        CHECK((static_cast<Fake&>(ladder[0]).values == std::vector<int>{128, 1}));
    }

    // Lists, without repeats or unusable items; the family's own variable
    // over the axis-wide one.
    setenv("OP_BLOCK_SIZES", "128,64,128,x", 1);
    setenv("OP_CHUNK_ATOMICS_UNSTAGED_SINGLES", "on,off", 1);
    setenv("OP_ATOMICS_BLOCK_SIZES", "256", 1);
    {
        loop::Ladder ladder{"test_loop"};
        ladder.add("chunk_atomics", loop::Default::enabled, axes, make);
        ladder.add("atomics", loop::Default::disabled, {axes.data(), 1}, make);
        ladder.add("colouring", loop::Default::enabled, {axes.data(), 1}, make);
        std::vector<std::string> names;
        for (std::size_t i = 0; i < ladder.size(); ++i)
            names.emplace_back(ladder[i].name());
        CHECK((names == std::vector<std::string>{
                            "chunk_atomics=on;block_size=64;singles=on",
                            "chunk_atomics=on;block_size=128;singles=off",
                            "chunk_atomics=on;block_size=64;singles=off",
                            "chunk_atomics=on;block_size=128;singles=on",
                            "atomics=on;block_size=256",
                            "colouring=on;block_size=64",
                            "colouring=on;block_size=128"}));
        for (std::size_t i : {0u, 1u, 2u, 4u, 5u})
            CHECK(!ladder.enabled(i) && ladder.explorable(i));
        CHECK(ladder.enabled(3) && ladder.enabled(6));
        CHECK(ladder[0].family() == "chunk_atomics");

        // A tuner's variant is never asked by the ladder.
        static_cast<Fake&>(ladder[3]).statuses = {loop::Status::rejected};
        CHECK(ladder.select(configuration()).strategy == &ladder[6]);
        for (std::size_t i : {0u, 1u, 2u, 4u, 5u})
            CHECK(static_cast<Fake&>(ladder[i]).prepares == 0);
    }
    unsetenv("OP_ATOMICS_BLOCK_SIZES");

    // OP_STRATEGY forces a family's running point; a variant's name forces
    // nothing.  OP_<FAMILY>=off turns every point off.
    setenv("OP_STRATEGY", "chunk_atomics,test_loop:colouring=on;block_size=64", 1);
    setenv("OP_COLOURING", "off", 1);
    loop::init();
    {
        loop::Ladder ladder{"test_loop"};
        ladder.add("chunk_atomics", loop::Default::disabled, axes, make);
        ladder.add("colouring", loop::Default::enabled, {axes.data(), 1}, make);
        CHECK(ladder.forced() == 3);
        CHECK(ladder.select(configuration()).strategy == &ladder[3]);
        CHECK(!ladder.explorable(4) && !ladder.explorable(5));
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
        auto *fake = add(ladder, "fake_" + std::to_string(i),
                         i % 2 == 0 ? loop::Default::enabled : loop::Default::disabled);
        fake->statuses.assign(4, loop::Status::rejected);
        fakes.push_back(fake);
    }
    fakes.back()->statuses.clear();
    for (int call = 0; call < 2; ++call) {
        auto selection = ladder.select(configuration());
        CHECK(selection.strategy == fakes.back());
        CHECK(selection.prepared.status == loop::Status::ready);
    }
    CHECK(fakes[0]->prepares == 2 && fakes[1]->prepares == 0);
}

void test_release_registry() {
    int releases = 0;
    auto release = [](void *owner) { ++*static_cast<int *>(owner); };

    loop::register_release(&releases, release);
    loop::release_all();
    CHECK(releases == 1);

    loop::unregister_release(&releases);
    loop::release_all();
    CHECK(releases == 1);
}

// The fast path: a select with the ladder's first strategy ready.
void time_selects() {
    Fixture fixture;
    auto config = configuration();
    constexpr int count = 1000000;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        auto selection = fixture.ladder.select(config);
        if (selection.strategy != fixture.atomics)
            std::abort();
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    std::printf("select: %.0f ns per call\n",
                std::chrono::duration<double, std::nano>(elapsed).count() / count);
}

} // namespace

int main() {
    try {
        test_configuration();
        test_defaults();
        test_variables();
        test_axes();
        test_long_ladder();
        test_release_registry();
        time_selects();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop ladder tests passed\n");
    return EXIT_SUCCESS;
}
