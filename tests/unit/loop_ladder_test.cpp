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
    // Statuses answered in turn; ready once exhausted.
    std::deque<loop::Status> statuses;
    std::string_view reason = "scripted";
    int prepares = 0;
    int now_prepares = 0;
    int runs = 0;
    float time = 1.0f;

    explicit Fake(std::string name) : fake_name{std::move(name)} {}

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

// A ladder of chunk_atomics (disabled), atomics, colouring (the floor).
struct Fixture {
    loop::Ladder ladder{"test_loop"};
    Fake *chunk_atomics;
    Fake *atomics;
    Fake *colouring;

    Fixture() {
        auto a = std::make_unique<Fake>("chunk_atomics");
        auto b = std::make_unique<Fake>("atomics");
        auto c = std::make_unique<Fake>("colouring");
        chunk_atomics = a.get();
        atomics = b.get();
        colouring = c.get();
        ladder.add(std::move(a), loop::Default::disabled);
        ladder.add(std::move(b), loop::Default::enabled);
        ladder.add(std::move(c), loop::Default::enabled);
    }
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
    CHECK(direct.wait_before(0) && !direct.wait_before(1));
    CHECK(direct.process_globals_after(0));
    CHECK(direct.set_stride() == 128);

    auto indirect = loop::Sections::indirect(&source, false);
    CHECK(indirect.size() == 2);
    CHECK(indirect[0].end == 100 && indirect[1].begin == 100 && indirect[1].end == 136);
    CHECK(!indirect.wait_before(0) && indirect.wait_before(1));
    CHECK(indirect.process_globals_after(1));
    CHECK(indirect.set_stride() == 160);

    auto separate = loop::Sections::indirect(&source, true);
    CHECK(separate.size() == 3);
    CHECK(separate[1].begin == 100 && separate[1].end == 128 && separate[2].end == 136);
    CHECK(separate.wait_before(1) && separate.process_globals_after(1));
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
    // OP_<FAMILY> enables or disables every variant of a family; a variant
    // registers disabled and the forced one starts the ladder.
    setenv("OP_CHUNK_ATOMICS", "1", 1);
    setenv("OP_ATOMICS", "0", 1);
    {
        loop::Ladder ladder{"test_loop"};
        ladder.add(std::make_unique<Fake>("chunk_atomics:256"), loop::Default::disabled);
        ladder.add(std::make_unique<Fake>("chunk_atomics"), loop::Default::disabled);
        ladder.add(std::make_unique<Fake>("atomics"), loop::Default::enabled);
        ladder.add(std::make_unique<Fake>("colouring"), loop::Default::enabled);
        CHECK(ladder.enabled(0) && ladder.enabled(1));
        CHECK(!ladder.enabled(2) && !ladder.explorable(2));
        CHECK(ladder.enabled(3));
        CHECK(ladder[0].family() == "chunk_atomics");

        auto selection = ladder.select(configuration());
        CHECK(selection.strategy->name() == "chunk_atomics:256");
    }
    unsetenv("OP_CHUNK_ATOMICS");
    unsetenv("OP_ATOMICS");

    // OP_STRATEGY forces one for every loop, or for a named loop, even when
    // its family is off; the ladder falls through from it.
    setenv("OP_STRATEGY", "other:colouring,test_loop:atomics:256", 1);
    loop::init();
    {
        loop::Ladder ladder{"test_loop"};
        ladder.add(std::make_unique<Fake>("chunk_atomics"), loop::Default::enabled);
        ladder.add(std::make_unique<Fake>("atomics:256"), loop::Default::disabled);
        ladder.add(std::make_unique<Fake>("atomics"), loop::Default::enabled);
        ladder.add(std::make_unique<Fake>("colouring"), loop::Default::enabled);
        CHECK(ladder.forced() == 1);
        auto selection = ladder.select(configuration());
        CHECK(selection.strategy->name() == "atomics:256");

        static_cast<Fake&>(ladder[1]).statuses = {loop::Status::pending};
        selection = ladder.select(configuration());
        CHECK(selection.strategy->name() == "atomics");
        CHECK(static_cast<Fake&>(ladder[0]).prepares == 0);
    }
    {
        loop::Ladder ladder{"other"};
        ladder.add(std::make_unique<Fake>("atomics"), loop::Default::enabled);
        ladder.add(std::make_unique<Fake>("colouring"), loop::Default::enabled);
        CHECK(ladder.forced() == 1);
        CHECK(ladder.select(configuration()).strategy->name() == "colouring");
    }
    {
        loop::Ladder ladder{"third"};
        ladder.add(std::make_unique<Fake>("atomics"), loop::Default::enabled);
        ladder.add(std::make_unique<Fake>("colouring"), loop::Default::enabled);
        CHECK(!ladder.forced().has_value());
    }
    unsetenv("OP_STRATEGY");
    loop::init();

    // A variant named by a word forces every loop that has it; a loop's own
    // item wins over it.
    setenv("OP_STRATEGY", "atomics:packed,other:colouring", 1);
    loop::init();
    for (const char *name : {"test_loop", "other"}) {
        loop::Ladder ladder{name};
        ladder.add(std::make_unique<Fake>("atomics:packed"), loop::Default::disabled);
        ladder.add(std::make_unique<Fake>("atomics"), loop::Default::enabled);
        ladder.add(std::make_unique<Fake>("colouring"), loop::Default::enabled);
        CHECK(ladder.forced() == (std::string_view{name} == "other" ? 2u : 0u));
    }
    unsetenv("OP_STRATEGY");
    loop::init();
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
        test_release_registry();
        time_selects();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop ladder tests passed\n");
    return EXIT_SUCCESS;
}
