#include <op2/rt/loop/loop.h>

#include <op2/rt/jit/report.h>
#include <op2/rt/report.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace op::rt::loop {

// --- Every loop ------------------------------------------------------------------

namespace {

// A loop and the descriptor that points at it.
struct Registered {
    std::unique_ptr<Loop> loop;
    abi::Loop *descriptor;
};

// Never destroyed: finalise empties it, and a program that skips op_exit
// must not free device state after the driver has shut down.
std::vector<Registered>& loops = *new std::vector<Registered>;
// Whether write_reports printed the summaries since the last finalise.
bool printed = false;

} // namespace

void register_loop(std::unique_ptr<Loop> loop, abi::Loop& descriptor) {
    loops.push_back({std::move(loop), &descriptor});
}

namespace {

std::vector<tune::TunedLoop> tuned() {
    std::vector<tune::TunedLoop> out;
    for (const auto& entry : loops)
        out.push_back({entry.loop->name(), entry.loop->tuner()});
    return out;
}

std::vector<jit::JitLoop> jitted() {
    std::vector<jit::JitLoop> out;
    for (const auto& entry : loops)
        out.push_back({entry.loop->name(), entry.loop->builds(), &entry.loop->params()});
    return out;
}

void write_files(int rank, int ranks) {
    std::vector<std::string> rows;
    for (const auto& entry : loops)
        if (const auto *tuner = entry.loop->tuner())
            tuner->rows(rows);
    report::write_csv(tune::autotune_options().report_file, rank, ranks,
                      tune::Autotuner::report_header, rows);

    rows.clear();
    for (const auto& loop : jitted())
        jit::rows(loop, rows);
    report::write_csv(jit::options().report_file, rank, ranks, jit::report_header, rows);
}

void print(int rank, int ranks) {
    printed = true;
    if (rank != 0)
        return;
    auto summaries = tune::summary(tuned(), ranks) + jit::summary(jitted(), ranks);
    std::fputs(summaries.c_str(), stdout);
    std::fflush(stdout);
}

} // namespace

void write_reports(int rank, int ranks) {
    write_files(rank, ranks);
    print(rank, ranks);
}

void finalise(int rank, int ranks) {
    write_files(rank, ranks);
    if (!printed)
        print(rank, ranks);
    for (auto& entry : loops)
        entry.descriptor->runtime = nullptr;
    loops.clear();
    printed = false;
}

} // namespace op::rt::loop
