#include <loop/autotune.h>
#include <op_env.h>

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

namespace op::loop {

// --- Options -------------------------------------------------------------------

namespace {

AutotuneOptions options;

} // namespace

const AutotuneOptions& autotune_options() { return options; }

void init_autotune_options() {
    options = {};
    constexpr env::Choice modes[] = {{"off", 0}, {"0", 0},      {"no", 0},
                                     {"false", 0}, {"report", 1}, {"select", 2}};
    int mode = 0;
    env::parse_choice("OP_AUTOTUNE", modes, mode);
    options.report = mode > 0;
    options.select = mode == 2;
    env::parse_text("OP_AUTOTUNE_REPORT", options.report_file);
    env::parse_number("OP_AUTOTUNE_MARGIN", options.margin);
    env::parse_number("OP_AUTOTUNE_WARMUP", options.warmup);
    env::parse_number("OP_AUTOTUNE_SAMPLES", options.samples, 1);
    env::parse_number("OP_AUTOTUNE_MIN_CALLS", options.min_calls);
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
    return !live ||
           (warmups >= options.warmup &&
            static_cast<int>(samples.size()) >= options.samples);
}

bool Autotuner::State::explored() const {
    if (trivial)
        return false;
    bool any = finished;
    for (const auto& trial : trials)
        any |= !trial.samples.empty();
    return any;
}

Autotuner::Decision Autotuner::choose(const Call& call,
                                      const Prepared& selection, bool settled) {
    Decision ladder{selection};
    auto key = call.key();
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
    if (state.finished)
        return run_selected(state, call, ladder);
    state.ladder = selection.candidate;
    if (state.calls < static_cast<std::size_t>(options.min_calls) || !settled)
        return ladder;
    if (state.trials.empty()) {
        add_trials(state, call, selection);
        if (state.trivial) {
            finish(state, call);
            return ladder;
        }
    }

    for (std::size_t step = 0; step < state.trials.size(); ++step) {
        std::size_t index = (state.cursor + step) % state.trials.size();
        auto& trial = state.trials[index];
        if (trial.done())
            continue;

        Prepared prepared = trial.candidate == selection.candidate
                                ? selection
                                : trial.candidate->prepare(call, Urgency::now);
        if (prepared.status == Status::pending)
            continue;
        if (prepared.status != Status::ready) {
            trial.live = false;
            trial.reason = prepared.reason;
            continue;
        }

        state.cursor = index + 1;
        Decision decision{prepared};
        if (trial.warmups < options.warmup)
            ++trial.warmups;
        else
            decision.timed = &trial;
        return decision;
    }

    for (const auto& trial : state.trials)
        if (!trial.done())
            return ladder;

    if (options.select)
        state.selected = winner(summarise(state));
    finish(state, call);
    return run_selected(state, call, ladder);
}

// An explored configuration: every strategy that ran or planned for it and
// that neither the ladder nor the selection can run again forgets it, so
// its plans go once no other strategy holds them.
void Autotuner::finish(State& state, const Call& call) {
    state.finished = true;
    auto kept = [&](const Strategy *strategy) {
        for (std::size_t i = 0; i < m_ladder.size(); ++i) {
            const auto& candidate = m_ladder[i];
            if (candidate.strategy == strategy &&
                (m_ladder.runs(candidate) || &candidate == state.selected))
                return true;
        }
        return false;
    };
    std::vector<Strategy *> forgotten;
    for (const auto& trial : state.trials) {
        auto *strategy = trial.candidate->strategy;
        if (trial.off || kept(strategy) ||
            std::find(forgotten.begin(), forgotten.end(), strategy) != forgotten.end())
            continue;
        strategy->forget(call.key());
        forgotten.push_back(strategy);
    }
}

// Every candidate not turned off is a trial: the ladder's choice, and every
// other whose prepare is ready or pending; the rest are retired with their
// reasons.  Trivial when the ladder's choice is the only one live.
void Autotuner::add_trials(State& state, const Call& call,
                           const Prepared& selection) {
    state.description = call.describe();
    int others = 0;
    for (std::size_t i = 0; i < m_ladder.size(); ++i) {
        Trial trial;
        trial.candidate = &m_ladder[i];
        if (trial.candidate->off) {
            trial.off = true;
        } else if (trial.candidate == selection.candidate) {
            trial.live = true;
        } else {
            auto prepared = trial.candidate->prepare(call, Urgency::later);
            trial.live = prepared.status == Status::ready ||
                         prepared.status == Status::pending;
            if (!trial.live)
                trial.reason = prepared.reason;
            else
                ++others;
        }
        state.trials.push_back(std::move(trial));
    }
    state.trivial = others == 0;
}

// What an explored configuration runs: its selected candidate when that is
// ready, else the ladder's choice.
Autotuner::Decision Autotuner::run_selected(const State& state,
                                            const Call& call,
                                            const Decision& ladder) {
    if (state.selected == nullptr || state.selected == ladder.prepared.candidate)
        return ladder;
    auto prepared = state.selected->prepare(call, Urgency::later);
    if (prepared.status != Status::ready)
        return ladder;
    return {prepared};
}

Autotuner::Summary Autotuner::summarise(const State& state) const {
    Summary summary;
    float fastest = 0.0f;
    for (const auto& trial : state.trials) {
        summary.complete &= trial.done();
        if (trial.candidate == state.ladder)
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

// The fastest candidate when it beats the ladder's choice by the margin;
// none, so the ladder's choice runs, otherwise.
const Candidate *Autotuner::winner(const Summary& summary) const {
    if (summary.fastest == nullptr || summary.fastest == summary.ladder)
        return nullptr;
    return summary.speedup * (1.0 - options.margin) > 1.0 ? summary.fastest->candidate
                                                          : nullptr;
}

void Autotuner::sample(const Decision& decision, float microseconds) {
    Trial *trial = decision.timed;
    if (trial == nullptr)
        return;
    trial->samples.push_back(microseconds);
    if (trial->done())
        trial->statistics = trial->candidate->strategy->statistics(decision.prepared);
}

// One candidate's report row for a configuration.
std::string Autotuner::row(const State& state, const Summary& summary,
                           const Trial& trial) const {
    std::string status = trial.live
                             ? (trial.samples.empty() ? "pending" : "timed")
                             : std::string{trial.reason};
    char median_text[32] = "";
    if (!trial.samples.empty())
        std::snprintf(median_text, sizeof(median_text), "%.1f",
                      median(trial.samples));
    char speedup_text[32];
    std::snprintf(speedup_text, sizeof(speedup_text), "%.3f", summary.speedup);
    const Candidate *selected = state.selected != nullptr ? state.selected : state.ladder;

    std::string text = m_ladder.name();
    for (const auto& column : {
             quoted(state.description), std::to_string(state.calls),
             trial.candidate->name, status,
             std::string{median_text}, std::to_string(trial.samples.size()),
             state.ladder->name,
             summary.fastest != nullptr ? summary.fastest->candidate->name : std::string{},
             std::string{speedup_text}, selected->name,
             std::string{summary.complete ? "complete" : "partial"},
             quoted(trial.statistics)})
        text += "," + column;
    return text;
}

void Autotuner::rows(std::vector<std::string>& out) const {
    for (const auto& [key, state] : m_states) {
        if (!state.explored())
            continue;
        auto summary = summarise(state);
        for (const auto& trial : state.trials)
            if (!trial.off)
                out.push_back(row(state, summary, trial));
    }
}

void Autotuner::outcomes(std::vector<Outcome>& out) const {
    for (const auto& [key, state] : m_states) {
        if (!state.explored())
            continue;
        auto summary = summarise(state);
        Outcome outcome{state.description, state.calls, state.ladder, nullptr,
                        state.selected, std::nullopt, std::nullopt, summary.complete};
        if (summary.ladder != nullptr && !summary.ladder->samples.empty())
            outcome.ladder_us = median(summary.ladder->samples);
        if (summary.fastest != nullptr) {
            outcome.fastest = summary.fastest->candidate;
            outcome.fastest_us = median(summary.fastest->samples);
        }
        out.push_back(outcome);
    }
}

std::size_t Autotuner::unexplored() const {
    std::size_t count = m_dropped;
    for (const auto& [key, state] : m_states)
        count += !state.explored() && !state.trivial;
    return count;
}

void Autotuner::clear() {
    m_states.clear();
    m_dropped = 0;
}

} // namespace op::loop
