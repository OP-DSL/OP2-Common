#pragma once

// The JIT's policy, as its variables set it: whether and when kernels are
// specialised, which parameters they bake in, and its report.

#include <optional>
#include <string>
#include <vector>

namespace op::rt::jit {

struct Options {
    // OP_JIT_LEVEL: 0 builds no specialised kernel; 1 bakes in the
    // structural parameters and the constants, a constant that takes too
    // many values moving to run time, and gives READ globals at run time; 2
    // also bakes in a READ global that takes few values.
    int level = 2;
    bool force = false;        // OP_JIT_FORCE: specialise every kernel
    bool seq_compile = false;  // OP_JIT_SEQ_COMPILE: build on the calling thread
    bool debug = false;        // OP_JIT_DEBUG: print builds and moves, sync launches
    // OP_JIT_MIN_CALLS: times a key is asked for before its build starts, so
    // values that change every call never build.
    int min_calls = 8;
    // OP_JIT_MAX_THREADS: builds under way at once on the run's build pool,
    // across every cache; the backend's default when unset.
    std::optional<int> max_threads;
    // OP_JIT_MAX_VALUES: values a baked parameter may take under a
    // configuration, the next moving it to run time for the rest of the run;
    // and the most a run-time READ global may take in 32 calls to be baked.
    int max_values = 8;
    // OP_JIT_MAX_GBL_BYTES, OP_JIT_MAX_CONST_BYTES: the largest READ global
    // and constant baked in; larger ones stay at run time.
    int max_gbl_bytes = 4096;
    int max_const_bytes = 16384;
    // OP_JIT_RUNTIME_PARAMS, OP_JIT_CONSTANT_PARAMS: parameters always given
    // at run time, or always baked in, by name ("<name>" or "<loop>:<name>").
    std::vector<std::string> runtime_params;
    std::vector<std::string> constant_params;
    // OP_JIT_REPORT: the report's file, ".<rank>" appended under MPI, and
    // the summary printed at the end; neither when empty.
    std::string report_file;

    bool enabled() const { return level > 0; }
};

// The policy the variables set; op::rt::init reads it into the run's
// context.
Options read_options();

} // namespace op::rt::jit
