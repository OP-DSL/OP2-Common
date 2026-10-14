#pragma once

// Strategy autotuning: each loop times its candidate strategies per
// configuration, and the results from every loop are written out at
// op_profile_output and op_exit.

#include <loop/strategy.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace op::loop {

// OP_AUTOTUNE=report times every candidate strategy for each loop
// configuration and reports how they compare with the ladder's choice, which
// still runs once a configuration has been explored.
struct AutotuneConfig {
    bool report = false;
    // OP_AUTOTUNE=select also reports, and once a configuration is explored
    // runs its fastest candidate whenever that beats the ladder's choice by
    // more than OP_AUTOTUNE_MARGIN, a fraction of the ladder's time.
    bool select = false;
    double margin = 0.03;
    // OP_AUTOTUNE_WARMUP untimed, then OP_AUTOTUNE_SAMPLES timed, runs of
    // each candidate.
    int warmup = 1;
    int samples = 3;
    // OP_AUTOTUNE_MIN_CALLS: calls a configuration must see before it is
    // explored, so configurations seen once are never explored.
    int min_calls = 2;
};

const AutotuneConfig& autotune_config();
void init_autotune_config();

// A loop's tuning: the timings of each candidate strategy for every
// configuration of the loop seen often enough, and what to run from them.
class Autotuner {
private:
    // One strategy's exploration for a configuration.
    struct Trial {
        Strategy *strategy = nullptr;
        bool candidate = false;
        // Why it is not a candidate, or was retired.
        std::string_view reason;
        int warmups = 0;
        std::vector<float> samples;
        std::string statistics;

        bool done() const;
    };

    // One configuration of the loop.
    struct State {
        std::string description;
        std::vector<Trial> trials;
        Strategy *ladder = nullptr;
        // What OP_AUTOTUNE=select runs once the configuration is explored.
        Strategy *selected = nullptr;
        std::size_t calls = 0;
        std::size_t cursor = 0;
        bool reported = false;
        // No candidate beyond the ladder's choice: nothing to explore.
        bool trivial = false;

        bool explored() const;
    };

public:
    // What one call runs: the strategy, its prepared value, and whether the
    // run is timed for the configuration being explored.
    struct Decision {
        Strategy *strategy = nullptr;
        Prepared prepared;
        bool timed = false;
        State *state = nullptr;
    };

private:
    const Ladder& m_ladder;
    // Configurations seen that often get no new state; the counts of those
    // and of configurations never explored are reported per loop.
    static constexpr std::size_t max_keys = 4096;
    std::unordered_map<std::uint64_t, State> m_states;
    std::size_t m_dropped = 0;

    Strategy *fastest_or_ladder(const State& state) const;
    std::string row(const State& state, const Trial& trial) const;
    std::string line(const State& state) const;
    void report(State& state);
    static void rows_callback(void *owner);

public:
    explicit Autotuner(const Ladder& ladder);
    ~Autotuner();
    Autotuner(const Autotuner&) = delete;

    // The ladder's selection, unless the configuration is being explored:
    // then each candidate (every strategy not turned off by its variable
    // whose prepare with now is ready) runs in turn for its warm-ups and
    // samples.  Pending defers a trial, unavailable or rejected retires it
    // with its reason.  When every trial is done the configuration is
    // reported and, under select, its fastest strategy runs from then on if
    // it beats the ladder's by the margin.
    Decision choose(const Configuration& configuration,
                    const Selection& selection);

    // Record one timed run.
    void sample(const Decision& decision, float microseconds);

    // At op_exit, print the configurations still being explored, and count
    // those never explored.
    void flush();
};

// A loop's tuning state, which adds a row per explored configuration and
// strategy with autotune_record whenever the report is written.
using AutotuneSource = void (*)(void *owner);
void autotune_register_source(void *owner, AutotuneSource add_rows);
void autotune_unregister_source(void *owner);

// Record one CSV row; header names the columns and is written once, ahead
// of the first row.
void autotune_record(std::string_view header, std::string row);

// Write every source's rows, as they stand, to OP_AUTOTUNE_REPORT, suffixed
// with the rank when there is more than one.  A later call overwrites the
// file, so applications that never call op_exit still get a report.
void autotune_write_report(int rank, int ranks);

} // namespace op::loop
