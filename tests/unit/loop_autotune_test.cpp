// The autotuner on the host: fake strategies with scripted statuses and
// timings for each of their kernels, explored per configuration and
// selected by the margin.

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/jit/report.h>
#include <op2/rt/lifecycle.h>
#include <op2/rt/loop/loop.h>
#include <op2/rt/tune/autotune.h>

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

#include "../test_support.h"

namespace loop = op::rt::loop;

namespace {

struct Fake : loop::Strategy {
    std::string fake_name;
    // Statuses answered in turn, ready once exhausted; jit_statuses for its
    // specialised kernel, which it has when jit_time is set.
    std::deque<loop::Status> statuses;
    std::deque<loop::Status> jit_statuses;
    std::string_view reason = "scripted";
    float time;
    std::optional<float> jit_time;
    int prepares = 0;
    int runs = 0;
    int timed_runs = 0;
    int timed_jit_runs = 0;
    int forgets = 0;
    // Its specialised kernels' builds, as it reports them.
    loop::BuildCounts fake_builds{};

    Fake(std::string name, float time_, std::optional<float> jit_time_ = {})
        : fake_name{std::move(name)}, time{time_}, jit_time{jit_time_} {}

    std::string_view name() const override { return fake_name; }

    // A specialised kernel when it has a jit_time, expected faster.
    loop::Estimate estimate(loop::KernelKind kind) const override {
        if (kind == loop::KernelKind::offline)
            return {.available = true, .cost = 1.0f};
        if (jit_time)
            return {.available = true, .cost = 0.9f};
        return {};
    }

    loop::Prepared prepare(const loop::Call&, loop::Urgency,
                           loop::KernelKind kind) override {
        ++prepares;
        auto& script = kind == loop::KernelKind::jit ? jit_statuses : statuses;
        loop::Status status = loop::Status::ready;
        if (!script.empty()) {
            status = script.front();
            script.pop_front();
        }
        loop::Prepared prepared;
        prepared.status = status;
        prepared.reason = status == loop::Status::ready ? "" : reason;
        return prepared;
    }

    std::optional<float> run(const loop::Prepared& prepared, const loop::Call&,
                             bool timed) override {
        CHECK(prepared.candidate->strategy == this);
        ++runs;
        if (!timed)
            return std::nullopt;
        if (prepared.candidate->kind == loop::KernelKind::jit) {
            ++timed_jit_runs;
            return *jit_time;
        }
        ++timed_runs;
        return time;
    }

    std::string statistics(const loop::Prepared& prepared) const override {
        return prepared.candidate->kind == loop::KernelKind::jit ? "jit" : "offline";
    }

    void forget(std::uint64_t) override { ++forgets; }

    loop::BuildCounts builds() const override { return fake_builds; }
};

// How a Fixture's fakes behave.
struct Settings {
    bool chunk_atomics_enabled = false;
    float chunk_atomics_time = 4.0f;
    std::optional<float> atomics_jit_time = {};
};

// A ladder of fakes and the tuner over it, as OP_AUTOTUNE says when they
// are built.
struct Fixture {
    loop::Ladder ladder{"flux", true};
    op::rt::tune::Autotuner tuner{ladder};
    Fake *chunk_atomics;
    Fake *atomics;
    Fake *chunk_colouring;
    Fake *colouring;
    // Whether the configurations' JIT modes have settled.
    bool settled = true;

    // A family without axes, declared now, its one strategy a Fake taking
    // time, and jit_time with its specialised kernel.
    Fake *add(std::string_view family, bool enabled, float time,
              std::optional<float> jit_time = {}) {
        const loop::Family declared[] = {{family, {}, enabled}};
        loop::declare_families(declared);
        Fake *fake = nullptr;
        ladder.add(family, [&](const loop::Point& point) {
            auto made = std::make_unique<Fake>(point.name, time, jit_time);
            fake = made.get();
            return made;
        });
        return fake;
    }

    explicit Fixture(const Settings& settings) {
        chunk_atomics = add("chunk_atomics", settings.chunk_atomics_enabled,
                            settings.chunk_atomics_time);
        atomics = add("atomics", true, 10.0f, settings.atomics_jit_time);
        chunk_colouring = add("chunk_colouring", false, 8.0f);
        colouring = add("colouring", true, 20.0f);
    }

    // The candidate the latest call ran.
    const loop::Candidate *last = nullptr;

    // One call: let the tuner decide, run, sample.
    Fake *invoke(const loop::Call& call) {
        auto decision = tuner.choose(call, settled);
        auto us = decision.prepared.candidate->strategy->run(decision.prepared, call,
                                                             decision.timed != nullptr);
        if (us)
            tuner.sample(decision, *us);
        last = decision.prepared.candidate;
        return static_cast<Fake *>(decision.prepared.candidate->strategy);
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
    op::rt::init();
    CHECK(op::rt::tune::autotune_options().select);

    Fixture fixture{Settings{}};
    loop::Call call{&edges, args, edges.size + edges.exec_size, false};

    // chunk_atomics compiles for two calls, chunk_colouring is turned off
    // and never asked.
    fixture.chunk_atomics->statuses = {loop::Status::pending, loop::Status::pending};

    // Seen once: the ladder's choice runs, untimed.
    CHECK(fixture.invoke(call) == fixture.atomics);
    CHECK(fixture.atomics->timed_runs == 0);

    // Explored from the second call: each candidate takes its warm-up and
    // two samples in turn, pending ones deferring, so nine calls explore
    // three candidates.
    for (int i = 0; i < 9; ++i)
        fixture.invoke(call);
    CHECK(fixture.chunk_colouring->runs == 0);
    CHECK(fixture.atomics->timed_runs == 2);
    CHECK(fixture.chunk_atomics->timed_runs == 2);
    CHECK(fixture.colouring->timed_runs == 2);
    CHECK(fixture.chunk_atomics->runs == 3);
    CHECK(fixture.colouring->runs == 3);
    CHECK(fixture.atomics->runs == 4);

    // Explored by the next call: the fastest runs from then on, untimed, and
    // the ladder no longer prepares its own choice.
    int before = fixture.chunk_atomics->runs;
    CHECK(fixture.invoke(call) == fixture.chunk_atomics);
    int ladder_prepares = fixture.atomics->prepares;
    for (int i = 0; i < 2; ++i)
        CHECK(fixture.invoke(call) == fixture.chunk_atomics);
    CHECK(fixture.chunk_atomics->runs == before + 3);
    CHECK(fixture.chunk_atomics->timed_runs == 2);
    CHECK(fixture.atomics->prepares == ladder_prepares);

    // A configuration on another set is explored on its own; there a
    // candidate that turns out unavailable is retired.
    loop::Call other{&cells, args, cells.size + cells.exec_size, false};
    fixture.chunk_atomics->statuses = {loop::Status::unavailable};
    fixture.chunk_atomics->reason = "no_kernel";
    before = fixture.atomics->timed_runs;
    for (int i = 0; i < 8; ++i)
        fixture.invoke(other);
    CHECK(fixture.atomics->timed_runs == before + 2);
    CHECK(fixture.chunk_atomics->timed_runs == 2);
    for (int i = 0; i < 2; ++i)
        CHECK(fixture.invoke(other) == fixture.atomics);

    // Once each configuration is explored, a strategy that neither the
    // ladder nor the selection runs forgets it: chunk_atomics on cells,
    // which it lost; on edges it was selected.
    CHECK(fixture.chunk_atomics->forgets == 1);
    CHECK(fixture.atomics->forgets == 0 && fixture.colouring->forgets == 0 &&
          fixture.chunk_colouring->forgets == 0);

    // The report: a row per configuration and strategy.
    std::vector<std::string> rows;
    fixture.tuner.rows(rows);
    CHECK(rows.size() == 6);
    CHECK(std::string_view{op::rt::tune::Autotuner::report_header}.rfind(
              "loop,configuration,calls,strategy,status", 0) == 0);
    int selected_chunk = 0;
    int retired = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        CHECK(rows[i].rfind("flux,", 0) == 0);
        CHECK(rows[i].find("chunk_colouring") == std::string::npos);
        selected_chunk += rows[i].find(",chunk_atomics=on,complete,") != std::string::npos;
        retired += rows[i].find(",no_kernel,") != std::string::npos;
    }
    CHECK(selected_chunk == 3);
    CHECK(retired == 1);

    // The summary: on edges chunk_atomics takes 4 to atomics' 10 over 13
    // calls, on cells atomics is fastest, so the loop takes 0.152 ms to the
    // ladder's 0.23.
    const op::rt::tune::TunedLoop tuned[] = {{"flux", &fixture.tuner}};
    auto summary = op::rt::tune::summary(tuned, 1);
    CHECK(summary.rfind("autotune summary:\n", 0) == 0);
    CHECK(summary.find("2 configurations of 1 loop explored; 0 not explored") !=
          std::string::npos);
    CHECK(summary.find("0.2 ms at the ladder's choices, 0.2 ms at the fastest (1.513x)") !=
          std::string::npos);
    CHECK(summary.find("other than the ladder's choice in 1 configuration\n") !=
          std::string::npos);
    CHECK(summary.find("  flux               2   1.513x       0.1  atomics  1 ") !=
          std::string::npos);
    CHECK(summary.find("1 chunk_atomics") != std::string::npos &&
          summary.find("1 atomics") != std::string::npos);
    CHECK(summary.find("JIT summary") == std::string::npos);
}

// The ladder's choice changes while a configuration is explored, once
// chunk_atomics' kernel arrives: select compares the fastest with the
// ladder's choice at the end, and keeps to the ladder unless it is beaten by
// the margin.
void test_ladder_changes() {
    setenv("OP_AUTOTUNE", "select", 1);
    setenv("OP_AUTOTUNE_MARGIN", "0.05", 1);
    setenv("OP_AUTOTUNE_WARMUP", "1", 1);
    setenv("OP_AUTOTUNE_SAMPLES", "2", 1);
    setenv("OP_AUTOTUNE_MIN_CALLS", "2", 1);
    setenv("OP_CHUNK_COLOURING", "0", 1);
    op::rt::init();

    Fixture fixture{{true, 10.2f}};
    loop::Call call{&edges, args, edges.size + edges.exec_size, false};
    fixture.chunk_atomics->statuses = {loop::Status::pending, loop::Status::pending,
                                       loop::Status::pending};
    CHECK(fixture.invoke(call) == fixture.atomics);
    for (int i = 0; i < 12; ++i)
        fixture.invoke(call);
    CHECK(fixture.atomics->timed_runs == 2 && fixture.chunk_atomics->timed_runs == 2);

    // atomics is fastest, but by less than the margin over chunk_atomics,
    // which the ladder now runs.
    for (int i = 0; i < 3; ++i)
        CHECK(fixture.invoke(call) == fixture.chunk_atomics);
}

// A strategy's two kernels are two candidates, each timed on its own: the
// specialised one is deferred while it builds, never timed on the offline
// kernel, and select can settle on either.  Nothing is explored before the
// configuration's JIT modes have settled.
void test_kernels() {
    setenv("OP_AUTOTUNE", "select", 1);
    setenv("OP_AUTOTUNE_MARGIN", "0", 1);
    setenv("OP_AUTOTUNE_WARMUP", "1", 1);
    setenv("OP_AUTOTUNE_SAMPLES", "2", 1);
    setenv("OP_AUTOTUNE_MIN_CALLS", "2", 1);
    setenv("OP_CHUNK_COLOURING", "0", 1);
    unsetenv("OP_AUTOTUNE_REPORT");
    op::rt::init();

    // atomics' specialised kernel, expected faster, comes first in the
    // ladder and takes 3 to atomics' offline 10.
    Fixture fixture{{false, 40.0f, 3.0f}};
    CHECK(fixture.ladder[1].name == "atomics=on;kernel=jit");
    CHECK(fixture.ladder[2].name == "atomics=on");
    loop::Call call{&edges, args, edges.size + edges.exec_size, false};

    // Unsettled, the ladder's choice runs untimed: the offline kernel while
    // the specialised one builds.
    fixture.settled = false;
    fixture.atomics->jit_statuses.assign(100, loop::Status::pending);
    for (int i = 0; i < 4; ++i)
        CHECK(fixture.invoke(call) == fixture.atomics &&
              fixture.last->kind == loop::KernelKind::offline);
    CHECK(fixture.atomics->timed_runs == 0 && fixture.atomics->timed_jit_runs == 0);

    // Settled, every other candidate takes its warm-up and two samples while
    // the specialised kernel is still pending, and that trial waits for it.
    fixture.settled = true;
    for (int i = 0; i < 9; ++i)
        fixture.invoke(call);
    CHECK(fixture.atomics->timed_runs == 2 && fixture.colouring->timed_runs == 2);
    CHECK(fixture.chunk_atomics->timed_runs == 2);
    CHECK(fixture.atomics->timed_jit_runs == 0);

    fixture.atomics->jit_statuses.clear();
    for (int i = 0; i < 3; ++i)
        fixture.invoke(call);
    CHECK(fixture.atomics->timed_jit_runs == 2);
    for (int i = 0; i < 2; ++i)
        CHECK(fixture.invoke(call) == fixture.atomics &&
              fixture.last->kind == loop::KernelKind::jit);
}

// A loop's specialised kernels and its configurations' modes, summarised and
// written to OP_JIT_REPORT, and neither without it: a constant that changes
// every call moves to run time on edges and stays baked in on cells.
void test_jit_report() {
    // The loop's descriptor, with the constant its executor registers in the
    // loop's parameters, and those parameters.
    struct Probe : op::rt::abi::Loop {
        double constant = 0.0;
        op::rt::jit::ParamSet *params = nullptr;
    };
    struct Jitted : loop::Executor {
        Jitted(Probe& probe, op::rt::jit::ParamSet& params) {
            params.add("op2_const_c_d", &probe.constant, op::rt::abi::ParamSource::external, -1,
                       true);
            probe.params = &params;
            const loop::Family declared[] = {{"atomics", {}}};
            loop::declare_families(declared);
        }
        // One strategy, whose builds the loop's are.
        void add_strategies(loop::Ladder& ladder) override {
            ladder.add("atomics", [](const loop::Point& point) {
                auto made = std::make_unique<Fake>(point.name, 1.0f, 1.0f);
                made->fake_builds = {3, 1, 0, 2};
                return made;
            });
        }
    };
    std::string path = "/tmp/op2_loop_jit_test_report.csv";
    unsetenv("OP_AUTOTUNE");
    unsetenv("OP_AUTOTUNE_REPORT");
    setenv("OP_JIT_REPORT", path.c_str(), 1);
    op::rt::init();

    Probe probe{{"flux_jit", "test", "Indirect", false}};
    auto& jitted = loop::runtime<Jitted>(probe);
    auto observe = [&](const loop::Call& call) {
        probe.params->update({});
        probe.params->observe_call(call);
    };
    loop::Call on_edges{&edges, args, edges.size + edges.exec_size, false};
    loop::Call on_cells{&cells, args, cells.size + cells.exec_size, false};
    for (int i = 0; i < 40; ++i) {
        probe.constant = i;
        observe(on_edges);
    }
    for (int i = 0; i < 10; ++i)
        observe(on_cells);

    auto jit_summary = [&] {
        const op::rt::jit::JitLoop loops[] = {
            {jitted.name(), jitted.builds(), &jitted.params()}};
        return op::rt::jit::summary(loops, 1);
    };
    auto summary = jit_summary();
    CHECK(summary.rfind("JIT summary:\n", 0) == 0);
    CHECK(summary.find("3 kernels built for 1 loop, 1 failed, 2 not built (asked for "
                       "fewer than 9 times)\n") != std::string::npos);
    CHECK(summary.find("1 loop with parameters that may move: 2 configurations, 1 "
                       "settled") != std::string::npos);
    CHECK(summary.find("  flux_jit  3 (1 failed)               2        1  "
                       "op2_const_c_d 1/1 (1)\n") != std::string::npos);

    loop::write_reports(0, 1);
    std::ifstream file(path);
    std::vector<std::string> rows;
    for (std::string line; std::getline(file, line);)
        rows.push_back(line);
    CHECK(rows.size() == 3);
    CHECK(rows[0] == "loop,configuration,calls,settled,baked,run_time,moved");
    int moved = 0, kept = 0;
    for (std::size_t i = 1; i < rows.size(); ++i) {
        moved += rows[i] == "flux_jit,\"edges (256)\",40,yes,,op2_const_c_d,op2_const_c_d";
        kept += rows[i] == "flux_jit,\"cells (64)\",10,no,op2_const_c_d,,";
    }
    CHECK(moved == 1 && kept == 1);
    std::remove(path.c_str());

    unsetenv("OP_JIT_REPORT");
    op::rt::init();
    CHECK(jit_summary().empty());
    loop::finalise(0, 1);
}

// The registry owns the loop made for a descriptor, and op_exit's finalise
// destroys it and clears the descriptor, so the next call makes another.
void test_loop_registry() {
    struct Counts : op::rt::abi::Loop {
        int destroyed = 0;
    };
    struct Counted : loop::Executor {
        Counts& counts;
        Counted(Counts& counts_, op::rt::jit::ParamSet&) : counts{counts_} {}
        ~Counted() override { ++counts.destroyed; }
        void add_strategies(loop::Ladder&) override {}
    };
    unsetenv("OP_AUTOTUNE_REPORT");
    op::rt::init();

    Counts counts{{"counted", "test", "Direct", true}};
    auto *first = &loop::runtime<Counted>(counts);
    CHECK(&loop::runtime<Counted>(counts) == first);
    loop::finalise(0, 1);
    CHECK(counts.destroyed == 1 && counts.runtime == nullptr);

    loop::runtime<Counted>(counts);
    loop::finalise(0, 1);
    CHECK(counts.destroyed == 2 && counts.runtime == nullptr);
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
        test_ladder_changes();
        test_kernels();
        test_jit_report();
        test_loop_registry();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("loop autotune tests passed\n");
    return EXIT_SUCCESS;
}
