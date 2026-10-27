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

// OP_AUTOTUNE=report times every candidate for each loop configuration and
// reports how they compare with the ladder's choice, which still runs once a
// configuration has been explored.
struct AutotuneOptions {
    bool report = false;
    // OP_AUTOTUNE=select also reports, and once a configuration is explored
    // runs its fastest candidate whenever that beats the ladder's current
    // choice by more than OP_AUTOTUNE_MARGIN, a fraction of the ladder's
    // time; otherwise the ladder's choice runs.
    bool select = false;
    double margin = 0.03;
    // OP_AUTOTUNE_WARMUP untimed, then OP_AUTOTUNE_SAMPLES timed, runs of
    // each candidate.
    int warmup = 1;
    int samples = 3;
    // OP_AUTOTUNE_MIN_CALLS: calls a configuration must see before it is
    // explored, so configurations seen once are never explored.
    int min_calls = 2;
    // OP_AUTOTUNE_REPORT: the report's file, ".<rank>" appended under MPI;
    // no report when empty.
    std::string report_file;
};

const AutotuneOptions& autotune_options();
// Read by init.
void init_autotune_options();

// A loop's tuning: the timings of each candidate for every configuration of
// the loop seen often enough, and what to run from them.
class Autotuner {
private:
    // One candidate's exploration for a configuration.
    struct Trial {
        const Candidate *candidate = nullptr;
        // Ready or pending when added, and not retired since.
        bool live = false;
        // Turned off by its variable: never explored or reported.
        bool off = false;
        // Why it is not live.
        std::string_view reason;
        int warmups = 0;
        std::vector<float> samples;
        std::string statistics;

        bool done() const;
    };

    // One configuration of the loop; its trials are added once it has been
    // seen min_calls times.
    struct State {
        std::string description;
        std::vector<Trial> trials;
        // The ladder's choice, as of the latest call while exploring.
        const Candidate *ladder = nullptr;
        // What OP_AUTOTUNE=select runs once the configuration is explored;
        // none to follow the ladder.
        const Candidate *selected = nullptr;
        std::size_t calls = 0;
        std::size_t cursor = 0;
        bool reported = false;
        // Nothing live beyond the ladder's choice: nothing to explore.
        bool trivial = false;

        bool explored() const;
    };

    // A configuration's trials against the ladder's choice.
    struct Summary {
        const Trial *fastest = nullptr;
        const Trial *ladder = nullptr;
        bool complete = true;
        // The ladder's median time over the fastest's; 0 without both.
        float speedup = 0.0f;
    };

public:
    // What one call runs, and the trial its run is a timed sample of while
    // the configuration is explored; none for an untimed run.
    struct Decision {
        Prepared prepared;
        Trial *timed = nullptr;
    };

private:
    const Ladder& m_ladder;
    // Once this many configurations have state a new one gets none; those
    // dropped and those never explored are counted per loop.
    static constexpr std::size_t max_keys = 4096;
    std::unordered_map<std::uint64_t, State> m_states;
    std::size_t m_dropped = 0;

    void add_trials(State& state, const Call& call,
                    const Prepared& selection);
    Decision run_selected(const State& state, const Call& call,
                          const Decision& ladder);
    Summary summarise(const State& state) const;
    const Candidate *winner(const Summary& summary) const;
    std::string row(const State& state, const Summary& summary,
                    const Trial& trial) const;
    std::string line(const State& state) const;
    void report(State& state);

public:
    explicit Autotuner(const Ladder& ladder) : m_ladder{ladder} {}
    Autotuner(const Autotuner&) = delete;

    // The ladder's selection, unless the configuration is being explored:
    // once seen min_calls times and settled (its JIT modes have had their
    // window), each candidate not turned off by its variable whose prepare
    // is ready or pending runs in turn for its warm-ups and samples.  Pending
    // defers a trial, its build started now; unavailable retires it with its
    // reason.  When every trial is done the configuration is reported and,
    // under select, its fastest candidate runs from then on if it beats the
    // ladder's current choice by the margin.
    Decision choose(const Call& call, const Prepared& selection, bool settled);

    // Record one timed run.
    void sample(const Decision& decision, float microseconds);

    // At op_exit, print the configurations still being explored, and count
    // those never explored.
    void flush();

    // Append a report row per explored configuration and strategy, as they
    // stand.
    void rows(std::vector<std::string>& out) const;
};

// Write every loop's report rows to OP_AUTOTUNE_REPORT, suffixed with the
// rank when there is more than one.  A later call overwrites the file, so
// applications that never call op_exit still get a report.
void autotune_write_report(int rank, int ranks);

// At op_exit, while the device is still up: write the report, then release
// every loop, which flushes the tuners the report reads.
void finalise(int rank, int ranks);

} // namespace op::loop
