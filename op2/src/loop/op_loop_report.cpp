#include <loop/loop.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace op::loop {

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

// --- Formatting ------------------------------------------------------------------

namespace {

void appendf(std::string& out, const char *format, ...) {
    char text[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    out += text;
}

// "1 loop", "2 loops".
std::string counted(std::size_t count, const char *noun) {
    return std::to_string(count) + " " + noun + (count == 1 ? "" : "s");
}

std::string quoted(std::string_view text) {
    std::string out = "\"";
    for (char c : text)
        out += c == '"' ? "\"\"" : std::string{c};
    return out + "\"";
}

// Labels and how many configurations each was found for: the label alone
// when there is one, else "2 colouring, 1 atomics jit", most first.
class Tally {
private:
    std::vector<std::pair<std::string, std::size_t>> m_counts;

public:
    void add(const std::string& label) {
        for (auto& [known, count] : m_counts)
            if (known == label) {
                ++count;
                return;
            }
        m_counts.emplace_back(label, 1);
    }

    std::string text() const {
        if (m_counts.size() == 1)
            return m_counts[0].first;
        auto sorted = m_counts;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string out;
        for (const auto& [label, count] : sorted)
            out += (out.empty() ? "" : ", ") + std::to_string(count) + " " + label;
        return out;
    }
};

// Rows of cells in columns two apart, indented by two; a column is as wide as
// its widest cell, and those in numeric are right-aligned.
std::string table(const std::vector<std::vector<std::string>>& rows,
                  const std::vector<bool>& numeric) {
    std::vector<std::size_t> widths(numeric.size(), 0);
    for (const auto& row : rows)
        for (std::size_t i = 0; i < row.size(); ++i)
            widths[i] = std::max(widths[i], row[i].size());
    std::string out;
    for (const auto& row : rows) {
        std::string line;
        for (std::size_t i = 0; i < row.size(); ++i) {
            std::string pad(widths[i] - row[i].size(), ' ');
            line += "  " + (numeric[i] ? pad + row[i] : row[i] + pad);
        }
        line.erase(line.find_last_not_of(' ') + 1);
        out += line + "\n";
    }
    return out;
}

// Loops listed by name past which the rest are collapsed into one line.
constexpr std::size_t listed = 20;

// --- The tuner's summary ----------------------------------------------------------

// A loop's explored configurations, and its time over their calls at the
// ladder's choices and at the fastest, for those with both timed.
struct LoopTuning {
    const Loop *loop;
    std::size_t configurations = 0;
    std::size_t partial = 0;
    std::size_t selected = 0;
    double ladder_ms = 0.0;
    double fastest_ms = 0.0;
    Tally ladder;
    Tally fastest;

    double saved_ms() const { return ladder_ms - fastest_ms; }
    // Beaten by more than a half percent.
    bool beaten() const { return ladder_ms > fastest_ms * 1.005; }
};

std::string tuner_summary(const std::vector<Loop *>& current, int ranks) {
    std::vector<LoopTuning> tunings;
    std::vector<Autotuner::Outcome> outcomes;
    std::size_t tuned = 0;
    std::size_t unexplored = 0;
    for (const auto *loop : current) {
        const auto *tuner = loop->tuner();
        if (tuner == nullptr)
            continue;
        ++tuned;
        unexplored += tuner->unexplored();
        outcomes.clear();
        tuner->outcomes(outcomes);
        if (outcomes.empty())
            continue;

        LoopTuning tuning{loop};
        for (const auto& outcome : outcomes) {
            ++tuning.configurations;
            tuning.partial += !outcome.complete;
            tuning.selected += outcome.selected != nullptr &&
                               outcome.selected != outcome.ladder;
            if (outcome.ladder != nullptr)
                tuning.ladder.add(outcome.ladder->label);
            if (outcome.fastest != nullptr)
                tuning.fastest.add(outcome.fastest->label);
            if (outcome.ladder_us && outcome.fastest_us) {
                double calls = static_cast<double>(outcome.calls);
                tuning.ladder_ms += calls * *outcome.ladder_us / 1000.0;
                tuning.fastest_ms += calls * *outcome.fastest_us / 1000.0;
            }
        }
        tunings.push_back(std::move(tuning));
    }
    if (tuned == 0)
        return {};

    std::string out = "autotune summary";
    if (ranks > 1)
        appendf(out, " (rank 0 of %d)", ranks);
    out += ":\n";
    std::size_t configurations = 0, partial = 0, selected = 0;
    double ladder_ms = 0.0, fastest_ms = 0.0;
    for (const auto& tuning : tunings) {
        configurations += tuning.configurations;
        partial += tuning.partial;
        selected += tuning.selected;
        ladder_ms += tuning.ladder_ms;
        fastest_ms += tuning.fastest_ms;
    }
    appendf(out, "  %s of %s explored", counted(configurations, "configuration").c_str(),
            counted(tunings.size(), "loop").c_str());
    if (partial > 0)
        appendf(out, " (%zu partly: a candidate was still pending)", partial);
    appendf(out, "; %zu not explored (seen fewer than %d times, or not settled)\n",
            unexplored, autotune_options().min_calls);
    if (configurations == 0)
        return out;
    appendf(out, "  over their calls: %.1f ms at the ladder's choices, %.1f ms at the "
                 "fastest (%.3fx)\n",
            ladder_ms, fastest_ms, fastest_ms > 0.0 ? ladder_ms / fastest_ms : 0.0);
    if (autotune_options().select)
        appendf(out, "  select runs a candidate other than the ladder's choice in %s\n",
                counted(selected, "configuration").c_str());

    auto end = std::partition(tunings.begin(), tunings.end(),
                              [](const auto& tuning) { return tuning.beaten(); });
    std::sort(tunings.begin(), end, [](const auto& a, const auto& b) {
        return a.saved_ms() > b.saved_ms();
    });
    std::size_t beaten = static_cast<std::size_t>(end - tunings.begin());
    if (beaten > 0) {
        std::vector<std::vector<std::string>> rows{
            {"loop", "configurations", "speedup", "saved_ms", "ladder", "fastest"}};
        double rest_ms = 0.0;
        for (std::size_t i = 0; i < beaten; ++i) {
            const auto& tuning = tunings[i];
            if (i >= listed) {
                rest_ms += tuning.saved_ms();
                continue;
            }
            std::string speedup, saved;
            appendf(speedup, "%.3fx", tuning.ladder_ms / tuning.fastest_ms);
            appendf(saved, "%.1f", tuning.saved_ms());
            rows.push_back({tuning.loop->name(), std::to_string(tuning.configurations),
                            speedup, saved, tuning.ladder.text(), tuning.fastest.text()});
        }
        out += table(rows, {false, true, true, true, false, false});
        if (beaten > listed)
            appendf(out, "  %s more where a candidate beat the ladder's choice, saving "
                         "%.1f ms\n",
                    counted(beaten - listed, "loop").c_str(), rest_ms);
    }
    if (tunings.size() > beaten)
        appendf(out, "  %s where the ladder's choice was the fastest, or within 0.5%%\n",
                counted(tunings.size() - beaten, "loop").c_str());
    return out;
}

// --- The JIT's summary --------------------------------------------------------------

// A loop's specialised kernels, and its configurations' modes: for each
// parameter that may move, in how many configurations it is baked in, at
// run time, and moved.
struct LoopJit {
    struct Counts {
        const Param *param;
        std::size_t baked = 0;
        std::size_t runtime = 0;
        std::size_t moved = 0;
    };

    const Loop *loop;
    BuildCounts builds;
    std::size_t configurations = 0;
    std::size_t settled = 0;
    std::vector<Counts> params;

    // Something to list: a parameter at run time or moved, or a failed
    // build.
    bool notable() const {
        if (builds.failed > 0)
            return true;
        return std::any_of(params.begin(), params.end(), [](const auto& counts) {
            return counts.runtime > 0 || counts.moved > 0;
        });
    }
};

LoopJit loop_jit(const Loop& loop) {
    LoopJit jit{&loop, loop.builds()};
    for (const auto& configuration : loop.params().configurations()) {
        ++jit.configurations;
        jit.settled += configuration.calls >= statistics_window;
        for (const auto& outcome : configuration.params) {
            auto found = std::find_if(jit.params.begin(), jit.params.end(),
                                      [&](const auto& counts) {
                                          return counts.param == outcome.param;
                                      });
            if (found == jit.params.end())
                found = jit.params.insert(found, LoopJit::Counts{outcome.param});
            ++(outcome.mode == ParamMode::baked ? found->baked : found->runtime);
            found->moved += outcome.moved;
        }
    }
    return jit;
}

std::string jit_summary(const std::vector<Loop *>& current, int ranks) {
    if (!jit_options().enabled() || jit_options().report_file.empty())
        return {};
    std::vector<LoopJit> jits;
    BuildCounts builds;
    std::size_t configurations = 0, settled = 0, moving = 0;
    for (const auto *loop : current) {
        auto jit = loop_jit(*loop);
        auto total = jit.builds.built + jit.builds.failed + jit.builds.building +
                     jit.builds.waiting;
        if (total == 0 && jit.configurations == 0)
            continue;
        builds += jit.builds;
        configurations += jit.configurations;
        settled += jit.settled;
        moving += !jit.params.empty();
        jits.push_back(std::move(jit));
    }
    if (jits.empty())
        return {};

    std::string out = "JIT summary";
    if (ranks > 1)
        appendf(out, " (rank 0 of %d)", ranks);
    out += ":\n";
    appendf(out, "  %s built for %s", counted(builds.built, "kernel").c_str(),
            counted(jits.size(), "loop").c_str());
    if (builds.failed > 0)
        appendf(out, ", %zu failed", builds.failed);
    if (builds.building > 0)
        appendf(out, ", %zu still building", builds.building);
    if (builds.waiting > 0)
        appendf(out, ", %zu not built (asked for fewer than %d times)", builds.waiting,
                jit_options().min_calls);
    out += "\n";
    if (moving == 0)
        return out;
    appendf(out, "  %s with parameters that may move: %s, %zu settled (observed for "
                 "%llu calls)\n",
            counted(moving, "loop").c_str(),
            counted(configurations, "configuration").c_str(), settled,
            static_cast<unsigned long long>(statistics_window));

    auto end = std::partition(jits.begin(), jits.end(),
                              [](const auto& jit) { return jit.notable(); });
    std::sort(jits.begin(), end, [](const auto& a, const auto& b) {
        return a.builds.built > b.builds.built;
    });
    std::size_t notable = static_cast<std::size_t>(end - jits.begin());
    if (notable > 0) {
        out += "  each parameter at run time or moved, in how many configurations it is "
               "baked in/at run time (moved):\n";
        std::vector<std::vector<std::string>> rows{
            {"loop", "kernels", "configurations", "settled", "parameters"}};
        for (std::size_t i = 0; i < std::min(notable, listed); ++i) {
            const auto& jit = jits[i];
            std::string kernels = std::to_string(jit.builds.built);
            if (jit.builds.failed > 0)
                appendf(kernels, " (%zu failed)", jit.builds.failed);
            std::string params;
            for (const auto& counts : jit.params) {
                if (counts.runtime == 0 && counts.moved == 0)
                    continue;
                params += params.empty() ? "" : ", ";
                appendf(params, "%s %zu/%zu", counts.param->name().c_str(), counts.baked,
                        counts.runtime);
                if (counts.moved > 0)
                    appendf(params, " (%zu)", counts.moved);
            }
            rows.push_back({jit.loop->name(), kernels, std::to_string(jit.configurations),
                            std::to_string(jit.settled), params});
        }
        out += table(rows, {false, true, true, true, false});
        if (notable > listed)
            appendf(out, "  %s more with a parameter at run time or moved, or a failed "
                         "build\n",
                    counted(notable - listed, "loop").c_str());
    }
    std::size_t baked = 0;
    for (auto jit = end; jit != jits.end(); ++jit)
        baked += !jit->params.empty();
    if (baked > 0)
        appendf(out, "  %s kept every parameter that may move baked in\n",
                counted(baked, "loop").c_str());
    return out;
}

// --- The files -----------------------------------------------------------------------

void write_csv(const std::string& path, int rank, int ranks, const char *header,
               const std::vector<std::string>& rows) {
    if (path.empty() || rows.empty())
        return;
    std::string name = path;
    if (ranks > 1)
        name += "." + std::to_string(rank);
    FILE *file = std::fopen(name.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "warning: cannot write report %s\n", name.c_str());
        return;
    }
    std::fprintf(file, "%s\n", header);
    for (const auto& row : rows)
        std::fprintf(file, "%s\n", row.c_str());
    std::fclose(file);
}

// A row per configuration whose parameters may move: the names baked in, at
// run time, and moved, each list separated by semicolons.
void jit_rows(const Loop& loop, std::vector<std::string>& out) {
    for (const auto& configuration : loop.params().configurations()) {
        std::string baked, runtime, moved;
        for (const auto& outcome : configuration.params) {
            auto& list = outcome.mode == ParamMode::baked ? baked : runtime;
            list += (list.empty() ? "" : ";") + outcome.param->name();
            if (outcome.moved)
                moved += (moved.empty() ? "" : ";") + outcome.param->name();
        }
        out.push_back(loop.name() + "," + quoted(configuration.description) + "," +
                      std::to_string(configuration.calls) + "," +
                      (configuration.calls >= statistics_window ? "yes" : "no") + "," +
                      baked + "," + runtime + "," + moved);
    }
}

void write_files(const std::vector<Loop *>& current, int rank, int ranks) {
    std::vector<std::string> rows;
    for (const auto *loop : current)
        if (const auto *tuner = loop->tuner())
            tuner->rows(rows);
    write_csv(autotune_options().report_file, rank, ranks, Autotuner::report_header, rows);

    rows.clear();
    for (const auto *loop : current)
        jit_rows(*loop, rows);
    write_csv(jit_options().report_file, rank, ranks,
              "loop,configuration,calls,settled,baked,run_time,moved", rows);
}

void print(const std::vector<Loop *>& current, int rank, int ranks) {
    printed = true;
    if (rank != 0)
        return;
    auto text = tuner_summary(current, ranks) + jit_summary(current, ranks);
    std::fputs(text.c_str(), stdout);
    std::fflush(stdout);
}

} // namespace

std::string summaries(int ranks) {
    auto current = current_loops();
    return tuner_summary(current, ranks) + jit_summary(current, ranks);
}

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

} // namespace op::loop
