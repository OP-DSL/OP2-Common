#include <op2/rt/tune/autotune.h>

#include <op2/rt/report.h>

#include <algorithm>
#include <string>
#include <vector>

namespace op::rt::tune {

using report::appendf;
using report::counted;
using report::listed;
using report::table;
using report::Tally;

namespace {

// A loop's explored configurations, and its time over their calls at the
// ladder's choices and at the fastest, for those with both timed.
struct LoopTuning {
    std::string_view name;
    std::size_t configurations = 0;
    std::size_t partial = 0;
    std::size_t selected = 0;
    double ladder_ms = 0.0;
    double fastest_ms = 0.0;
    Tally ladder{};
    Tally fastest{};

    double saved_ms() const { return ladder_ms - fastest_ms; }
    // Beaten by more than a half percent.
    bool beaten() const { return ladder_ms > fastest_ms * 1.005; }
};

} // namespace

std::string summary(std::span<const TunedLoop> loops, const AutotuneOptions& options,
                    int ranks) {
    std::vector<LoopTuning> tunings;
    std::vector<Autotuner::Outcome> outcomes;
    std::size_t tuned = 0;
    std::size_t unexplored = 0;
    for (const auto& loop : loops) {
        const auto *tuner = loop.tuner;
        if (tuner == nullptr)
            continue;
        ++tuned;
        unexplored += tuner->unexplored();
        outcomes.clear();
        tuner->outcomes(outcomes);
        if (outcomes.empty())
            continue;

        LoopTuning tuning{loop.name};
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
            unexplored, options.min_calls);
    if (configurations == 0)
        return out;
    appendf(out, "  over their calls: %.1f ms at the ladder's choices, %.1f ms at the "
                 "fastest (%.3fx)\n",
            ladder_ms, fastest_ms, fastest_ms > 0.0 ? ladder_ms / fastest_ms : 0.0);
    if (options.select)
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
            rows.push_back({std::string{tuning.name}, std::to_string(tuning.configurations),
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

} // namespace op::rt::tune
