#pragma once

// The JIT's report: its summary of every loop's specialised kernels and
// parameter modes, and a CSV row per configuration whose parameters may
// move.  Defined in src/rt/jit/op_jit_report.cpp.

#include <op2/rt/jit/params.h>
#include <op2/rt/loop/strategy.h>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::jit {

// A loop as the report reads it: its kernels' builds and its parameters.
struct JitLoop {
    std::string_view name;
    loop::BuildCounts builds;
    const ParamSet *params;
};

// The summary printed under OP_JIT_REPORT when the loops built or observed
// anything; empty otherwise.
std::string summary(std::span<const JitLoop> loops, int ranks);

// The report's columns, and a row per configuration of loop whose parameters
// may move: the names baked in, at run time, and moved, each list separated
// by semicolons.
constexpr const char *report_header = "loop,configuration,calls,settled,baked,run_time,moved";
void rows(const JitLoop& loop, std::vector<std::string>& out);

} // namespace op::rt::jit
