#include <op2/rt/tune/autotune.h>

#include <op2/rt/report.h>

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

namespace op::rt::tune {

// --- Per-loop tuning --------------------------------------------------------------

namespace {

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    auto middle = values.size() / 2;
    return values.size() % 2 != 0
               ? values[middle]
               : 0.5f * (values[middle - 1] + values[middle]);
}

} // namespace

bool Autotuner::Trial::done(const AutotuneOptions& options) const {
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

Autotuner::Decision Autotuner::choose(const loop::Call& call, bool settled) {
    auto key = call.key();
    auto found = m_states.find(key);
    if (found != m_states.end() && found->second.finished) {
        ++found->second.calls;
        return run_selected(found->second, call, std::nullopt);
    }

    auto selection = m_ladder.select(call);
    Decision ladder{selection};
    if (found == m_states.end()) {
        if (m_states.size() >= max_keys) {
            ++m_dropped;
            return ladder;
        }
        found = m_states.emplace(key, State{}).first;
    }

    auto& state = found->second;
    ++state.calls;
    state.ladder = selection.candidate;
    if (state.calls < static_cast<std::size_t>(m_options.min_calls) || !settled)
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
        if (trial.done(m_options))
            continue;

        loop::Prepared prepared = trial.candidate == selection.candidate
                                ? selection
                                : trial.candidate->prepare(call, loop::Urgency::now);
        if (prepared.status == loop::Status::pending)
            continue;
        if (prepared.status != loop::Status::ready) {
            trial.live = false;
            trial.reason = prepared.reason;
            continue;
        }

        state.cursor = index + 1;
        Decision decision{prepared};
        if (trial.warmups < m_options.warmup)
            ++trial.warmups;
        else
            decision.timed = &trial;
        return decision;
    }

    for (const auto& trial : state.trials)
        if (!trial.done(m_options))
            return ladder;

    if (m_options.select)
        state.selected = winner(summarise(state));
    finish(state, call);
    return run_selected(state, call, selection);
}

// An explored configuration: every strategy that ran or planned for it and
// that neither the ladder nor the selection can run again forgets it, so
// its plans go once no other strategy holds them.
void Autotuner::finish(State& state, const loop::Call& call) {
    state.finished = true;
    auto kept = [&](const loop::Strategy *strategy) {
        for (std::size_t i = 0; i < m_ladder.size(); ++i) {
            const auto& candidate = m_ladder[i];
            if (candidate.strategy == strategy &&
                (m_ladder.runs(candidate) || &candidate == state.selected))
                return true;
        }
        return false;
    };
    std::vector<loop::Strategy *> forgotten;
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
void Autotuner::add_trials(State& state, const loop::Call& call,
                           const loop::Prepared& selection) {
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
            auto prepared = trial.candidate->prepare(call, loop::Urgency::later);
            trial.live = prepared.status == loop::Status::ready ||
                         prepared.status == loop::Status::pending;
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
// ready, else the ladder's choice, asked for unless the call has it.
Autotuner::Decision Autotuner::run_selected(const State& state, const loop::Call& call,
                                            const std::optional<loop::Prepared>& ladder) {
    if (state.selected != nullptr && (!ladder || state.selected != ladder->candidate)) {
        auto prepared = state.selected->prepare(call, loop::Urgency::later);
        if (prepared.status == loop::Status::ready)
            return {prepared};
    }
    return {ladder ? *ladder : m_ladder.select(call)};
}

Autotuner::Summary Autotuner::summarise(const State& state) const {
    Summary summary;
    float fastest = 0.0f;
    for (const auto& trial : state.trials) {
        summary.complete &= trial.done(m_options);
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
const loop::Candidate *Autotuner::winner(const Summary& summary) const {
    if (summary.fastest == nullptr || summary.fastest == summary.ladder)
        return nullptr;
    return summary.speedup * (1.0 - m_options.margin) > 1.0 ? summary.fastest->candidate
                                                          : nullptr;
}

void Autotuner::sample(const Decision& decision, float microseconds) {
    Trial *trial = decision.timed;
    if (trial == nullptr)
        return;
    trial->samples.push_back(microseconds);
    if (trial->done(m_options))
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
    const loop::Candidate *selected = state.selected != nullptr ? state.selected : state.ladder;

    std::string text = m_ladder.name();
    for (const auto& column : {
             report::quoted(state.description), std::to_string(state.calls),
             trial.candidate->name, status,
             std::string{median_text}, std::to_string(trial.samples.size()),
             state.ladder->name,
             summary.fastest != nullptr ? summary.fastest->candidate->name : std::string{},
             std::string{speedup_text}, selected->name,
             std::string{summary.complete ? "complete" : "partial"},
             report::quoted(trial.statistics)})
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
        Outcome outcome{.description = state.description,
                        .calls = state.calls,
                        .ladder = state.ladder,
                        .fastest = nullptr,
                        .selected = state.selected,
                        .ladder_us = std::nullopt,
                        .fastest_us = std::nullopt,
                        .complete = summary.complete};
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

} // namespace op::rt::tune
