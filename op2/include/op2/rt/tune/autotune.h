#pragma once

// Strategy autotuning: each loop times its candidates per configuration;
// op2/rt/loop/loop.h's write_reports summarises every loop's results and
// writes them out at op_profile_output and op_exit.

#include <op2/rt/loop/call.h>
#include <op2/rt/loop/ladder.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/tune/options.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace op::rt::tune {

// A loop's tuning: the timings of each candidate for every configuration of
// the loop seen often enough, and what to run from them.
class Autotuner {
    struct Trial;

public:
    // The report's columns.
    static constexpr const char *report_header =
        "loop,configuration,calls,strategy,status,median_us,samples,ladder,"
        "fastest,speedup,selected,state,statistics";

    // What one call runs, and the trial its run is a timed sample of while
    // the configuration is explored; none for an untimed run.
    struct Decision {
        loop::Prepared prepared;
        Trial *timed = nullptr;
    };

    // An explored configuration, for the summary: the ladder's choice and
    // the fastest candidate with their median times, none until timed.
    struct Outcome {
        std::string_view description;
        std::size_t calls;
        const loop::Candidate *ladder;
        const loop::Candidate *fastest;
        const loop::Candidate *selected;
        std::optional<float> ladder_us;
        std::optional<float> fastest_us;
        bool complete;
    };

    // The tuner of ladder's loop under options, which must outlive it.
    Autotuner(loop::Ladder& ladder, const AutotuneOptions& options)
        : m_ladder{ladder}, m_options{options} {}
    Autotuner(const Autotuner&) = delete;

    // The ladder's selection, unless the configuration is being explored:
    // once seen min_calls times and settled (its JIT modes have had their
    // window), each candidate not turned off by its variable whose prepare
    // is ready or pending runs in turn for its warm-ups and samples.  Pending
    // defers a trial, its build started now; unavailable retires it with its
    // reason.  When every trial is done, under select, its fastest
    // candidate runs from then on if it beats the ladder's current choice by
    // the margin, the ladder not asked while it is ready.
    Decision choose(const loop::Call& call, bool settled);

    // Record one timed run.
    void sample(const Decision& decision, float microseconds);

    void outcomes(std::vector<Outcome>& out) const;
    // Configurations seen with something to explore but not explored,
    // those past max_keys included.
    std::size_t unexplored() const;

    // Append a report row per explored configuration and candidate, as they
    // stand.
    void rows(std::vector<std::string>& out) const;

private:
    // One candidate's exploration for a configuration.
    struct Trial {
        const loop::Candidate *candidate = nullptr;
        // Ready or pending when added, and not retired since.
        bool live = false;
        // Turned off by its variable: never explored or in the report.
        bool off = false;
        // Why it is not live.
        std::string_view reason;
        int warmups = 0;
        std::vector<float> samples;
        std::string statistics;

        bool done(const AutotuneOptions& options) const;
    };

    // One configuration of the loop; its trials are added once it has been
    // seen min_calls times.
    struct State {
        std::string description;
        std::vector<Trial> trials;
        // The ladder's choice, as of the latest call while exploring.
        const loop::Candidate *ladder = nullptr;
        // What OP_AUTOTUNE=select runs once the configuration is explored;
        // none to follow the ladder.
        const loop::Candidate *selected = nullptr;
        std::size_t calls = 0;
        std::size_t cursor = 0;
        // Every trial is done.
        bool finished = false;
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

    // Asked for its selection when a call needs it.
    loop::Ladder& m_ladder;
    const AutotuneOptions& m_options;
    // Once this many configurations have state a new one gets none; those
    // dropped and those never explored are counted per loop.
    static constexpr std::size_t max_keys = 4096;
    std::unordered_map<std::uint64_t, State> m_states;
    std::size_t m_dropped = 0;

    void add_trials(State& state, const loop::Call& call,
                    const loop::Prepared& selection);
    // Mark the configuration explored, and have the strategies nothing
    // will run on it again forget it.
    void finish(State& state, const loop::Call& call);
    Decision run_selected(const State& state, const loop::Call& call,
                          const std::optional<loop::Prepared>& ladder);
    Summary summarise(const State& state) const;
    const loop::Candidate *winner(const Summary& summary) const;
    std::string row(const State& state, const Summary& summary,
                    const Trial& trial) const;
};

// A loop as the summary reads it: its tuner, none when it has none.
struct TunedLoop {
    std::string_view name;
    const Autotuner *tuner;
};

// The summary of the loops' tuning, empty when none was tuned.
std::string summary(std::span<const TunedLoop> loops, const AutotuneOptions& options,
                    int ranks);

} // namespace op::rt::tune
