#include <op2/rt/loop/registry.h>

#include <op2/rt/jit/report.h>
#include <op2/rt/loop/loop.h>
#include <op2/rt/report.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace op::rt::loop {

// --- Every loop ------------------------------------------------------------------

Registry::Registry() = default;

Registry::~Registry() {
    for (auto& entry : m_loops)
        entry.descriptor->runtime = nullptr;
}

void Registry::add(std::unique_ptr<Loop> loop, abi::Loop& descriptor) {
    m_loops.push_back({std::move(loop), &descriptor});
}

void Registry::write_reports(const Context& context, int rank, int ranks, bool end) {
    std::vector<tune::TunedLoop> tuned;
    std::vector<jit::JitLoop> jitted;
    for (const auto& entry : m_loops) {
        tuned.push_back({entry.loop->name(), entry.loop->tuner()});
        jitted.push_back({entry.loop->name(), entry.loop->builds(), &entry.loop->params()});
    }

    std::vector<std::string> rows;
    for (const auto& entry : m_loops)
        if (const auto *tuner = entry.loop->tuner())
            tuner->rows(rows);
    report::write_csv(context.autotune.report_file, rank, ranks,
                      tune::Autotuner::report_header, rows);
    rows.clear();
    for (const auto& loop : jitted)
        jit::rows(loop, rows);
    report::write_csv(context.jit.report_file, rank, ranks, jit::report_header, rows);

    if (end && m_printed)
        return;
    m_printed = true;
    if (rank != 0)
        return;
    auto summaries = tune::summary(tuned, context.autotune, ranks) +
                     jit::summary(jitted, context.jit, ranks);
    std::fputs(summaries.c_str(), stdout);
    std::fflush(stdout);
}

} // namespace op::rt::loop
