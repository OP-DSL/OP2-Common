#include <loop/autotune.h>
#include <op_lib_c.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace op::loop {

// --- Report -----------------------------------------------------------------------

namespace {

std::mutex tuners_mutex;
std::vector<const Autotuner *> tuners;

constexpr const char *header =
    "loop,configuration,calls,strategy,status,median_us,samples,ladder,"
    "fastest,speedup,selected,state,statistics";

} // namespace

void autotune_write_report(int rank, int ranks) {
    const char *path = std::getenv("OP_AUTOTUNE_REPORT");
    if (path == nullptr || path[0] == '\0')
        return;

    std::vector<std::string> rows;
    {
        std::scoped_lock lock(tuners_mutex);
        for (const auto *tuner : tuners)
            tuner->rows(rows);
    }
    if (rows.empty())
        return;

    std::string name = path;
    if (ranks > 1)
        name += "." + std::to_string(rank);

    FILE *file = std::fopen(name.c_str(), "w");
    if (file == nullptr) {
        std::fprintf(stderr, "warning: cannot write autotune report %s\n",
                     name.c_str());
        return;
    }

    std::fprintf(file, "%s\n", header);
    for (const auto& row : rows)
        std::fprintf(file, "%s\n", row.c_str());
    std::fclose(file);
}

void finalise(int rank, int ranks) {
    autotune_write_report(rank, ranks);
    release_all();
}

// --- Configuration --------------------------------------------------------------

namespace {

AutotuneConfig config;

} // namespace

const AutotuneConfig& autotune_config() { return config; }

void init_autotune_config() {
    config = {};
    if (const char *text = std::getenv("OP_AUTOTUNE")) {
        auto value = env::lower(text);
        if (value == "report" || value == "select") {
            config.report = true;
            config.select = value == "select";
        } else if (value != "0" && value != "off" && value != "no" &&
                   value != "false") {
            std::fprintf(stderr,
                         "warning: ignoring OP_AUTOTUNE='%s', expected 0, "
                         "report or select\n",
                         text);
        }
    }
    env::parse_number("OP_AUTOTUNE_MARGIN", config.margin);
    env::parse_number("OP_AUTOTUNE_WARMUP", config.warmup);
    env::parse_number("OP_AUTOTUNE_SAMPLES", config.samples);
    env::parse_number("OP_AUTOTUNE_MIN_CALLS", config.min_calls);
    config.samples = std::max(config.samples, 1);
}

// --- Per-loop tuning --------------------------------------------------------------

namespace {

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    auto middle = values.size() / 2;
    return values.size() % 2 != 0
               ? values[middle]
               : 0.5f * (values[middle - 1] + values[middle]);
}

std::string quoted(std::string_view text) {
    std::string out = "\"";
    for (char c : text)
        out += c == '"' ? "\"\"" : std::string{c};
    return out + "\"";
}

} // namespace

bool Autotuner::Trial::done() const {
    return !candidate ||
           (warmups >= config.warmup &&
            static_cast<int>(samples.size()) >= config.samples);
}

bool Autotuner::State::explored() const {
    if (trivial)
        return false;
    bool any = reported;
    for (const auto& trial : trials)
        any |= !trial.samples.empty();
    return any;
}

Autotuner::Autotuner(const Ladder& ladder) : m_ladder{ladder} {
    std::scoped_lock lock(tuners_mutex);
    tuners.push_back(this);
}

Autotuner::~Autotuner() {
    std::scoped_lock lock(tuners_mutex);
    std::erase(tuners, this);
}

Autotuner::Decision Autotuner::choose(const Configuration& configuration,
                                      const Prepared& selection) {
    Decision ladder{selection};
    auto key = configuration.key();
    auto found = m_states.find(key);
    if (found == m_states.end()) {
        if (m_states.size() >= max_keys) {
            ++m_dropped;
            return ladder;
        }
        found = m_states.emplace(key, State{}).first;
    }

    auto& state = found->second;
    ++state.calls;
    if (state.reported)
        return run_selected(state, configuration, ladder);
    state.ladder = selection.strategy;
    if (state.calls < static_cast<std::size_t>(config.min_calls))
        return ladder;
    if (state.trials.empty()) {
        add_trials(state, configuration, selection);
        if (state.trivial) {
            state.reported = true;
            return ladder;
        }
    }

    for (std::size_t step = 0; step < state.trials.size(); ++step) {
        std::size_t index = (state.cursor + step) % state.trials.size();
        auto& trial = state.trials[index];
        if (trial.done())
            continue;

        Prepared prepared = trial.strategy == selection.strategy
                                ? selection
                                : trial.strategy->prepare(configuration, Urgency::now);
        if (prepared.status == Status::pending)
            continue;
        if (prepared.status != Status::ready) {
            trial.candidate = false;
            trial.reason = prepared.reason;
            continue;
        }

        state.cursor = index + 1;
        Decision decision{prepared};
        if (trial.warmups < config.warmup)
            ++trial.warmups;
        else
            decision.timed = &trial;
        return decision;
    }

    for (const auto& trial : state.trials)
        if (!trial.done())
            return ladder;

    if (config.select)
        state.selected = winner(summarise(state));
    report(state);
    return run_selected(state, configuration, ladder);
}

// Every strategy not turned off is a trial: the ladder's choice, and every
// other whose prepare is ready or pending; the rest are retired with their
// reasons.  Trivial when the ladder's choice is the only candidate.
void Autotuner::add_trials(State& state, const Configuration& configuration,
                           const Prepared& selection) {
    state.description = configuration.describe();
    int others = 0;
    for (std::size_t i = 0; i < m_ladder.size(); ++i) {
        Trial trial;
        trial.strategy = const_cast<Strategy *>(&m_ladder[i]);
        if (!m_ladder.explorable(i)) {
            trial.reason = "disabled";
        } else if (trial.strategy == selection.strategy) {
            trial.candidate = true;
        } else {
            auto prepared = trial.strategy->prepare(configuration, Urgency::later);
            trial.candidate = prepared.status == Status::ready ||
                              prepared.status == Status::pending;
            if (!trial.candidate)
                trial.reason = prepared.reason;
            else
                ++others;
        }
        state.trials.push_back(std::move(trial));
    }
    state.trivial = others == 0;
}

// What an explored configuration runs: its selected strategy when that is
// ready, else the ladder's choice.
Autotuner::Decision Autotuner::run_selected(const State& state,
                                            const Configuration& configuration,
                                            const Decision& ladder) {
    if (state.selected == nullptr || state.selected == ladder.prepared.strategy)
        return ladder;
    auto prepared = state.selected->prepare(configuration, Urgency::later);
    if (prepared.status != Status::ready)
        return ladder;
    return {prepared};
}

Autotuner::Summary Autotuner::summarise(const State& state) const {
    Summary summary;
    float fastest = 0.0f;
    for (const auto& trial : state.trials) {
        summary.complete &= trial.done();
        if (trial.strategy == state.ladder)
            summary.ladder = &trial;
        if (trial.samples.empty())
            continue;
        float time = median(trial.samples);
        if (summary.fastest == nullptr || time < fastest) {
            summary.fastest = &trial;
            fastest = time;
        }
    }
    if (summary.fastest != nullptr && summary.ladder != nullptr &&
        !summary.ladder->samples.empty())
        summary.speedup = median(summary.ladder->samples) / fastest;
    return summary;
}

// The fastest strategy when it beats the ladder's choice by the margin;
// none, so the ladder's choice runs, otherwise.
Strategy *Autotuner::winner(const Summary& summary) const {
    if (summary.fastest == nullptr || summary.fastest == summary.ladder)
        return nullptr;
    return summary.speedup * (1.0 - config.margin) > 1.0 ? summary.fastest->strategy
                                                          : nullptr;
}

void Autotuner::sample(const Decision& decision, float microseconds) {
    Trial *trial = decision.timed;
    if (trial == nullptr)
        return;
    trial->samples.push_back(microseconds);
    if (trial->done())
        trial->statistics = trial->strategy->statistics(decision.prepared);
}

// One strategy's report row for a configuration.
std::string Autotuner::row(const State& state, const Summary& summary,
                           const Trial& trial) const {
    std::string status = trial.candidate
                             ? (trial.samples.empty() ? "pending" : "timed")
                             : std::string{trial.reason};
    char median_text[32] = "";
    if (!trial.samples.empty())
        std::snprintf(median_text, sizeof(median_text), "%.1f",
                      median(trial.samples));
    char speedup_text[32];
    std::snprintf(speedup_text, sizeof(speedup_text), "%.3f", summary.speedup);
    Strategy *selected = state.selected != nullptr ? state.selected : state.ladder;

    std::string text = m_ladder.name();
    for (const auto& column : {
             quoted(state.description), std::to_string(state.calls),
             std::string{trial.strategy->name()}, status,
             std::string{median_text}, std::to_string(trial.samples.size()),
             std::string{state.ladder->name()},
             summary.fastest != nullptr ? std::string{summary.fastest->strategy->name()}
                                        : std::string{},
             std::string{speedup_text}, std::string{selected->name()},
             std::string{summary.complete ? "complete" : "partial"},
             quoted(trial.statistics)})
        text += "," + column;
    return text;
}

// The printed line: each strategy's median time, or why it was not a
// candidate, against the ladder's choice.
std::string Autotuner::line(const State& state) const {
    std::string times;
    for (const auto& trial : state.trials) {
        if (!trial.candidate && trial.reason == "disabled")
            continue;

        times += times.empty() ? "" : ", ";
        times += std::string{trial.strategy->name()} + " ";
        if (!trial.candidate) {
            times += trial.reason;
        } else if (trial.samples.empty()) {
            times += "pending";
        } else {
            char text[32];
            std::snprintf(text, sizeof(text), "%.1fus", median(trial.samples));
            times += text;
            if (!trial.statistics.empty())
                times += " (" + trial.statistics + ")";
        }
    }

    auto summary = summarise(state);
    char tail[160];
    std::snprintf(tail, sizeof(tail), "; ladder %s, fastest %s (%.2fx)%s",
                  std::string{state.ladder->name()}.c_str(),
                  summary.fastest != nullptr
                      ? std::string{summary.fastest->strategy->name()}.c_str()
                      : "none",
                  summary.speedup, summary.complete ? "" : ", partial");
    std::string text = "autotune: " + m_ladder.name() + " on " +
                       state.description + ": " + times + tail;
    if (state.selected != nullptr)
        text += ", selected " + std::string{state.selected->name()};
    return text;
}

// Print a configuration's results on the root rank, once.
void Autotuner::report(State& state) {
    if (state.reported)
        return;
    state.reported = true;
    if (op_is_root())
        std::printf("%s\n", line(state).c_str());
}

void Autotuner::rows(std::vector<std::string>& out) const {
    for (const auto& [key, state] : m_states) {
        if (!state.explored())
            continue;
        auto summary = summarise(state);
        for (const auto& trial : state.trials)
            if (trial.candidate || trial.reason != "disabled")
                out.push_back(row(state, summary, trial));
    }
}

void Autotuner::flush() {
    std::size_t unexplored = m_dropped;
    for (auto& [key, state] : m_states) {
        if (state.explored())
            report(state);
        else if (!state.trivial)
            ++unexplored;
    }

    if (unexplored > 0 && op_is_root())
        std::printf("autotune: %s left %zu configurations unexplored "
                    "(seen fewer than %d times, or past %zu)\n",
                    m_ladder.name().c_str(), unexplored, config.min_calls,
                    max_keys);
    m_states.clear();
    m_dropped = 0;
}

} // namespace op::loop
