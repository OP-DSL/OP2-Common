#include <op2/rt/loop/ladder.h>

#include <op2/env.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <utility>

namespace op::rt::loop {

// --- Strategies ---------------------------------------------------------------

Prepared Candidate::prepare(const Call& call, Urgency urgency) const {
    Prepared prepared;
    if (!refused.empty())
        prepared.reason = refused;
    else
        prepared = strategy->prepare(call, urgency, kind);
    prepared.candidate = this;
    return prepared;
}

// --- The ladder ---------------------------------------------------------------

namespace {

// The family, and the axis values where point differs from first:
// "chunk_atomics (block_size=64 singles=off)".
std::string label(std::string_view family, std::span<const Axis> axes, const Point& point,
                  const Point& first) {
    std::string differing;
    for (std::size_t i = 0; i < point.values.size(); ++i)
        if (point.values[i] != first.values[i])
            differing += (differing.empty() ? "" : " ") + axes[i].item(point.values[i]);
    std::string text{family};
    return differing.empty() ? text : text + " (" + differing + ")";
}

} // namespace

Ladder::Ladder(std::string_view loop_name, bool explore, const Families& families,
               const jit::Options& jit)
    : m_name{loop_name}, m_explore{explore}, m_families{families}, m_jit{jit},
      m_forced_names{families.forced_for(loop_name)} {}

void Ladder::add(std::string_view family, const Make& make) {
    const auto *found = m_families.find(family);
    if (found == nullptr) {
        std::fprintf(stderr,
                     "error: %s registers strategy family %s, which no backend declared\n",
                     m_name.c_str(), std::string{family}.c_str());
        std::exit(1);
    }
    const auto& settings = *found;
    bool enabled = settings.enabled.value_or(settings.enabled_by_default);
    bool off = !enabled && settings.enabled.has_value();

    auto points = settings.points(family);
    const auto& first = points.front();
    // A strategy at point, member when the ladder runs it, and its candidates.
    auto add_point = [&](const Point& point, bool member) {
        auto& strategy = *m_strategies.emplace_back(make(point));
        for (const auto& other : m_strategies)
            if (other.get() != &strategy && other->name() == strategy.name()) {
                std::fprintf(stderr, "error: strategy %s registered twice (in %s)\n",
                             std::string{strategy.name()}.c_str(), m_name.c_str());
                std::exit(1);
            }

        auto add = [&](KernelKind kind, std::string_view refused = {}) {
            std::string name{strategy.name()};
            std::string short_name = label(family, settings.axes, point, first);
            if (kind == KernelKind::jit) {
                name += ";kernel=jit";
                short_name += " jit";
            }
            m_candidates.push_back({.strategy = &strategy,
                                    .kind = kind,
                                    .name = std::move(name),
                                    .label = std::move(short_name),
                                    .family = std::string{family},
                                    .rank = settings.rank,
                                    .member = member,
                                    .enabled = member && enabled,
                                    .off = off,
                                    .refused = refused});
        };
        auto offline = strategy.estimate(KernelKind::offline);
        auto jit = m_jit.enabled() ? strategy.estimate(KernelKind::jit) : Estimate{};
        bool jit_first = jit.available &&
                         (!offline.available || m_jit.force ||
                          (jit.cost && offline.cost && *jit.cost < *offline.cost));
        if (jit_first)
            add(KernelKind::jit);
        if (offline.available)
            add(KernelKind::offline);
        if (jit.available && !jit_first)
            add(KernelKind::jit);
        // With neither kernel, only the ladder's point keeps a candidate,
        // refused, to say why.
        if (!offline.available && !jit.available && member)
            add(KernelKind::offline,
                !jit.reason.empty()       ? jit.reason
                : !offline.reason.empty() ? offline.reason
                                          : std::string_view{"kernel_unavailable"});
    };
    // The family's other points, when the ladder explores, ahead of its
    // point of first values.
    if (m_explore)
        for (std::size_t i = 1; i < points.size(); ++i)
            add_point(points[i], false);
    add_point(first, true);
    order();
}

// Ladder order, stable within a family, and the forced family's members.
void Ladder::order() {
    m_order.resize(m_candidates.size());
    std::iota(m_order.begin(), m_order.end(), std::size_t{0});
    std::stable_sort(m_order.begin(), m_order.end(), [&](std::size_t a, std::size_t b) {
        return m_candidates[a].rank < m_candidates[b].rank;
    });
    m_reasons.assign(m_order.size(), std::string_view{});

    m_forced.reset();
    for (const auto& family : m_forced_names) {
        for (std::size_t index = 0; index < size(); ++index)
            if ((*this)[index].member && (*this)[index].family == family) {
                if (!m_forced)
                    m_forced = index;
                m_forced_last = index;
            }
        if (m_forced)
            return;
    }
}

Prepared Ladder::select(const Call& call) {
    assert(size() > 0);
    std::fill(m_reasons.begin(), m_reasons.end(), std::string_view{});

    Prepared selection;
    for (std::size_t index = m_forced.value_or(0); index < size(); ++index) {
        const auto& candidate = (*this)[index];
        // The tuner's other points are no part of the ladder.
        if (!candidate.member)
            continue;
        bool forced = of_forced_family(candidate);
        if (!candidate.enabled && !forced) {
            m_reasons[index] = "disabled";
            continue;
        }
        // The run asked for the forced family, so its kernels' builds start
        // at once and its last kernel is waited for.
        bool wait = forced && index == m_forced_last;
        selection = candidate.prepare(call, wait     ? Urgency::wait
                                            : forced ? Urgency::now
                                                     : Urgency::later);
        if (selection.status == Status::ready) {
            report(selection, {m_reasons.data(), index});
            return selection;
        }
        m_reasons[index] = selection.reason;
        if (wait)
            warn_forced(selection.reason);
    }

    // The floor must run, so wait for its kernel; without one the loop
    // cannot run at all.
    const auto& floor = (*this)[size() - 1];
    selection = floor.prepare(call, Urgency::wait);
    if (selection.status != Status::ready) {
        std::fprintf(stderr, "error: %s has nothing to run: %s is %s\n",
                     m_name.c_str(), floor.name.c_str(),
                     std::string{selection.reason}.c_str());
        std::exit(1);
    }
    report(selection, {m_reasons.data(), size() - 1});
    return selection;
}

// A family's rank is its ID in the run.
bool Ladder::of_forced_family(const Candidate& candidate) const {
    return m_forced && candidate.rank == (*this)[*m_forced].rank;
}

bool Ladder::runs(const Candidate& candidate) const {
    return (candidate.member && (candidate.enabled || of_forced_family(candidate))) ||
           &candidate == &(*this)[size() - 1];
}

void Ladder::warn_forced(std::string_view reason) {
    if (std::find(m_warned.begin(), m_warned.end(), reason) != m_warned.end())
        return;

    m_warned.emplace_back(reason);
    env::warn("warning: " + m_name + " passes over its forced strategy " +
              (*this)[m_forced_last].name + ": " + m_warned.back() + "\n");
}

void Ladder::report(const Prepared& selection,
                    std::span<const std::string_view> reasons) {
    if (OP_diags <= 3)
        return;

    std::string passed;
    for (std::size_t i = 0; i < reasons.size(); ++i) {
        if (reasons[i].empty())
            continue;
        passed += passed.empty() ? " (" : ", ";
        passed += (*this)[i].name;
        passed += ": ";
        passed += reasons[i];
    }
    if (!passed.empty())
        passed += ")";
    std::string outcome = selection.candidate->name + passed;
    if (std::find(m_reported.begin(), m_reported.end(), outcome) !=
        m_reported.end())
        return;

    m_reported.push_back(outcome);
    op_printf("strategy: %s runs %s\n", m_name.c_str(), outcome.c_str());
}

} // namespace op::rt::loop
