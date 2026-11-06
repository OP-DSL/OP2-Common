#pragma once

// The tuner's run-wide options, which op::rt::init reads into the run's
// context.

#include <string>

namespace op::rt::tune {

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
    int samples = 5;
    // OP_AUTOTUNE_MIN_CALLS: calls a configuration must see before it is
    // explored, so configurations seen once are never explored.
    int min_calls = 2;
    // OP_AUTOTUNE_REPORT: the report's file, ".<rank>" appended under MPI;
    // no report when empty.
    std::string report_file;
};

// The tuner's options the variables set; op::rt::init reads them into the
// run's context.  Defined in src/rt/tune/op_tune_options.cpp.
AutotuneOptions read_options();

} // namespace op::rt::tune
