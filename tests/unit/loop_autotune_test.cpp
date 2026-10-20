// The autotuner on the host: fake strategies with scripted statuses and
// timings, explored per configuration and selected by the margin.

#include <loop/autotune.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
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
    std::deque<loop::Status> statuses;
    std::string_view reason = "scripted";
    float time;
    int runs = 0;
    int timed_runs = 0;

    Fake(std::string name, float time_) : fake_name{std::move(name)}, time{time_} {}

    std::string_view name() const override { return fake_name; }

    loop::Prepared prepare(const loop::Configuration&, loop::Urgency) override {
        loop::Status status = loop::Status::ready;
        if (!statuses.empty()) {
            status = statuses.front();
            statuses.pop_front();
        }
        loop::Prepared prepared;
        prepared.strategy = this;
        prepared.status = status;
        prepared.reason = status == loop::Status::ready ? "" : reason;
        return prepared;
    }

    std::optional<float> run(const loop::Prepared& prepared,
                             const loop::Launch& launch) override {
        CHECK(prepared.strategy == this);
        ++runs;
        if (!launch.timed)
            return std::nullopt;
        ++timed_runs;
        return time;
    }

    std::string statistics(const loop::Prepared&) const override {
        return "jit";
    }
};

struct Fixture {
    loop::Ladder ladder{"flux"};
    Fake *chunk_atomics;
    Fake *atomics;
    Fake *chunk_colouring;
    Fake *colouring;
    loop::Autotuner tuner{ladder};

    // A family without axes, its one strategy a Fake taking time.
    Fake *add(std::string_view family, loop::Default setting, float time) {
        Fake *fake = nullptr;
        ladder.add(family, setting, {}, [&](const loop::Point& point) {
            auto made = std::make_unique<Fake>(point.name, time);
            fake = made.get();
            return made;
        });
        return fake;
    }

    Fixture() {
        chunk_atomics = add("chunk_atomics", loop::Default::disabled, 4.0f);
        atomics = add("atomics", loop::Default::enabled, 10.0f);
        chunk_colouring = add("chunk_colouring", loop::Default::disabled, 8.0f);
        colouring = add("colouring", loop::Default::enabled, 20.0f);
    }

    // One call: select, let the tuner decide, run, sample.
    Fake *call(const loop::Configuration& configuration) {
        auto selection = ladder.select(configuration);
        auto decision = tuner.choose(configuration, selection);
        loop::Launch launch{const_cast<op_arg *>(configuration.args.data()),
                            static_cast<int>(configuration.args.size()),
                            decision.timed};
        auto us = decision.strategy->run(decision.prepared, launch);
        if (us)
            tuner.sample(decision, *us);
        return static_cast<Fake *>(decision.strategy);
    }
};

op_set_core edges{};
op_set_core cells{};
std::array<op_arg, 1> args{};

void test_exploration() {
    setenv("OP_AUTOTUNE", "select", 1);
    setenv("OP_AUTOTUNE_MARGIN", "0", 1);
    setenv("OP_AUTOTUNE_WARMUP", "1", 1);
    setenv("OP_AUTOTUNE_SAMPLES", "2", 1);
    setenv("OP_AUTOTUNE_MIN_CALLS", "2", 1);
    setenv("OP_CHUNK_COLOURING", "0", 1);
    loop::init();
    CHECK(loop::autotune_config().select);

    Fixture fixture;
    loop::Configuration configuration{&edges, args};

    // chunk_atomics compiles for two calls, chunk_colouring is turned off
    // and never asked.
    fixture.chunk_atomics->statuses = {loop::Status::pending, loop::Status::pending};

    // Seen once: the ladder's choice runs, untimed.
    CHECK(fixture.call(configuration) == fixture.atomics);
    CHECK(fixture.atomics->timed_runs == 0);

    // Explored from the second call: each candidate takes its warm-up and
    // two samples in turn, pending ones deferring, so nine calls explore
    // three candidates.
    for (int i = 0; i < 9; ++i)
        fixture.call(configuration);
    CHECK(fixture.chunk_colouring->runs == 0);
    CHECK(fixture.atomics->timed_runs == 2);
    CHECK(fixture.chunk_atomics->timed_runs == 2);
    CHECK(fixture.colouring->timed_runs == 2);
    CHECK(fixture.chunk_atomics->runs == 3);
    CHECK(fixture.colouring->runs == 3);
    CHECK(fixture.atomics->runs == 4);

    // Explored: the fastest runs from now on, untimed.
    int before = fixture.chunk_atomics->runs;
    for (int i = 0; i < 3; ++i)
        CHECK(fixture.call(configuration) == fixture.chunk_atomics);
    CHECK(fixture.chunk_atomics->runs == before + 3);
    CHECK(fixture.chunk_atomics->timed_runs == 2);

    // A configuration on another set is explored on its own; there a
    // candidate that turns out unavailable is retired.
    loop::Configuration other{&cells, args};
    fixture.chunk_atomics->statuses = {loop::Status::unavailable};
    fixture.chunk_atomics->reason = "no_kernel";
    before = fixture.atomics->timed_runs;
    for (int i = 0; i < 8; ++i)
        fixture.call(other);
    CHECK(fixture.atomics->timed_runs == before + 2);
    CHECK(fixture.chunk_atomics->timed_runs == 2);
    for (int i = 0; i < 2; ++i)
        CHECK(fixture.call(other) == fixture.atomics);

    // The report: a row per configuration and strategy.
    std::string path = "/tmp/op2_loop_autotune_test_report.csv";
    setenv("OP_AUTOTUNE_REPORT", path.c_str(), 1);
    loop::autotune_write_report(0, 1);
    std::ifstream file(path);
    std::vector<std::string> rows;
    for (std::string line; std::getline(file, line);)
        rows.push_back(line);
    CHECK(rows.size() == 7);
    CHECK(rows[0].rfind("loop,configuration,calls,strategy,status", 0) == 0);
    int selected_chunk = 0;
    int retired = 0;
    for (std::size_t i = 1; i < rows.size(); ++i) {
        CHECK(rows[i].rfind("flux,", 0) == 0);
        selected_chunk += rows[i].find(",chunk_atomics=on,complete,") != std::string::npos;
        retired += rows[i].find(",no_kernel,") != std::string::npos;
    }
    CHECK(selected_chunk == 3);
    CHECK(retired == 1);
    std::remove(path.c_str());

    fixture.tuner.flush();
}

} // namespace

int main() {
    edges.index = 0;
    edges.size = 256;
    edges.core_size = 256;
    edges.name = "edges";
    cells.index = 1;
    cells.size = 64;
    cells.core_size = 64;
    cells.name = "cells";
    args[0].opt = 1;
    args[0].argtype = OP_ARG_DAT;

    try {
        test_exploration();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop autotune tests passed\n");
    return EXIT_SUCCESS;
}
