#include <op2/rt/loop/loop.h>

#include <op2/rt/jit/report.h>
#include <op2/rt/report.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace op::rt::loop {

// --- Every loop ------------------------------------------------------------------

namespace {

std::mutex loops_mutex;
std::vector<Loop *> loops;
// Whether write_reports printed the summaries since the last finalise.
bool printed = false;

std::vector<Loop *> current_loops() {
    std::scoped_lock lock(loops_mutex);
    return loops;
}

} // namespace

void register_loop(Loop *loop) {
    std::scoped_lock lock(loops_mutex);
    loops.push_back(loop);
}

void unregister_loop(Loop *loop) {
    std::scoped_lock lock(loops_mutex);
    std::erase(loops, loop);
}

namespace {

std::vector<tune::TunedLoop> tuned(const std::vector<Loop *>& current) {
    std::vector<tune::TunedLoop> out;
    for (const auto *loop : current)
        out.push_back({loop->name(), loop->tuner()});
    return out;
}

std::vector<jit::JitLoop> jitted(const std::vector<Loop *>& current) {
    std::vector<jit::JitLoop> out;
    for (const auto *loop : current)
        out.push_back({loop->name(), loop->builds(), &loop->params()});
    return out;
}

std::string text(const std::vector<Loop *>& current, int ranks) {
    return tune::summary(tuned(current), ranks) + jit::summary(jitted(current), ranks);
}

void write_files(const std::vector<Loop *>& current, int rank, int ranks) {
    std::vector<std::string> rows;
    for (const auto *loop : current)
        if (const auto *tuner = loop->tuner())
            tuner->rows(rows);
    report::write_csv(tune::autotune_options().report_file, rank, ranks,
                      tune::Autotuner::report_header, rows);

    rows.clear();
    for (const auto& loop : jitted(current))
        jit::rows(loop, rows);
    report::write_csv(jit::options().report_file, rank, ranks, jit::report_header, rows);
}

void print(const std::vector<Loop *>& current, int rank, int ranks) {
    printed = true;
    if (rank != 0)
        return;
    auto summaries = text(current, ranks);
    std::fputs(summaries.c_str(), stdout);
    std::fflush(stdout);
}

} // namespace

void write_reports(int rank, int ranks) {
    auto current = current_loops();
    write_files(current, rank, ranks);
    print(current, rank, ranks);
}

void finalise(int rank, int ranks) {
    auto current = current_loops();
    write_files(current, rank, ranks);
    if (!printed)
        print(current, rank, ranks);
    for (auto *loop : current)
        loop->release();
    printed = false;
}

} // namespace op::rt::loop
