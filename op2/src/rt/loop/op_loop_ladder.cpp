#include <op2/rt/loop/ladder.h>

#include <op2/env.h>
#include <op2/rt/jit/options.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
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

// --- Variants -----------------------------------------------------------------

int Point::value(std::string_view key) const {
    auto found = std::find(keys.begin(), keys.end(), key);
    assert(found != keys.end());
    return values[static_cast<std::size_t>(found - keys.begin())];
}

namespace {

// OP_STRATEGY's items, as init read them.
std::vector<std::string> strategy_items;

// A declared family's axes, default, OP_<FAMILY>, each axis's values, and
// its place in ladder order.
struct FamilySettings {
    std::vector<Axis> axes;
    bool enabled_by_default;
    std::optional<bool> enabled;
    std::vector<std::vector<int>> values;
    std::size_t rank;
};

std::map<std::string, FamilySettings, std::less<>> family_settings;
std::size_t declared_families = 0;

// The families forced on a loop, most specific first: "<loop>:<family>"
// forces that loop, a bare family every loop that has it.
std::vector<std::string> forced_for(std::string_view loop) {
    std::vector<std::string> specific;
    std::vector<std::string> every;
    for (std::string_view item : strategy_items) {
        if (item.size() > loop.size() && item.substr(0, loop.size()) == loop &&
            item[loop.size()] == ':')
            specific.emplace_back(item.substr(loop.size() + 1));
        else if (item.find(':') == std::string_view::npos)
            every.emplace_back(item);
    }
    specific.insert(specific.end(), every.begin(), every.end());
    return specific;
}

std::string upper(std::string_view text) {
    std::string out{text};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return out;
}

// An axis's values from one variable, in order without repeats; empty when
// the variable is unset or holds nothing usable.
std::vector<int> axis_values(const std::string& variable, const Axis& axis) {
    std::vector<std::string> items;
    env::parse_list(variable.c_str(), items);
    std::vector<int> values;
    for (const auto& item : items) {
        int value = -1;
        if (axis.binary) {
            if (auto parsed = env::switch_value(item))
                value = *parsed ? 1 : 0;
        } else {
            char *end = nullptr;
            long parsed = std::strtol(item.c_str(), &end, 10);
            if (end != item.c_str() && *end == '\0' && parsed > 0 && parsed <= INT32_MAX)
                value = static_cast<int>(parsed);
        }
        if (value < 0)
            env::warn("warning: ignoring " + variable + " item '" + item +
                      "', expected " + (axis.binary ? "on or off" : "a positive number") +
                      "\n");
        else if (std::find(values.begin(), values.end(), value) == values.end())
            values.push_back(value);
    }
    return values;
}

// Each axis's values for family: its own variable's, else the axis-wide
// variable's, else the defaults.
std::vector<std::vector<int>> family_values(std::string_view family,
                                            std::span<const Axis> axes) {
    std::vector<std::vector<int>> values;
    for (const auto& axis : axes) {
        auto variable = std::string{axis.variable};
        auto chosen = axis_values("OP_" + upper(family) + "_" + variable, axis);
        if (chosen.empty())
            chosen = axis_values("OP_" + variable, axis);
        if (chosen.empty())
            chosen = axis.defaults;
        assert(!chosen.empty());
        values.push_back(std::move(chosen));
    }
    return values;
}

// Every point of the values, the first values' first, named
// "<family>=on;<key>=<value>;...", a switch's value on or off.
std::vector<Point> points(std::string_view family, std::span<const Axis> axes,
                          const std::vector<std::vector<int>>& values) {
    std::vector<Point> out;
    std::vector<std::size_t> index(axes.size(), 0);
    while (true) {
        Point point{std::string{family} + "=on", {}, {}};
        for (std::size_t i = 0; i < axes.size(); ++i) {
            int value = values[i][index[i]];
            point.keys.push_back(axes[i].key);
            point.values.push_back(value);
            point.name += ";" + std::string{axes[i].key} + "=" +
                          (axes[i].binary ? (value != 0 ? "on" : "off")
                                          : std::to_string(value));
        }
        out.push_back(std::move(point));

        std::size_t axis = 0;
        for (; axis < axes.size(); ++axis) {
            if (++index[axis] < values[axis].size())
                break;
            index[axis] = 0;
        }
        if (axis == axes.size())
            return out;
    }
}

// The family, and the axis values where point differs from first:
// "chunk_atomics (block_size=64 singles=off)".
std::string label(std::string_view family, const Point& point, const Point& first) {
    std::string differing;
    std::string_view items = point.name;
    for (std::size_t i = 0; i < point.values.size(); ++i) {
        items.remove_prefix(items.find(';') + 1);
        if (point.values[i] != first.values[i])
            differing += (differing.empty() ? "" : " ") +
                         std::string{items.substr(0, items.find(';'))};
    }
    std::string text{family};
    return differing.empty() ? text : text + " (" + differing + ")";
}

} // namespace

void init() {
    strategy_items.clear();
    env::parse_list("OP_STRATEGY", strategy_items);
    family_settings.clear();
    declared_families = 0;
}

void declare_families(std::span<const Family> families) {
    for (const auto& family : families) {
        FamilySettings settings;
        settings.axes = family.axes;
        settings.enabled_by_default = family.enabled;
        env::parse_bool(("OP_" + upper(family.name)).c_str(), settings.enabled);
        settings.values = family_values(family.name, family.axes);
        settings.rank = declared_families++;
        family_settings.insert_or_assign(std::string{family.name}, std::move(settings));
    }
    for (std::string_view item : strategy_items) {
        auto family = item.substr(item.find(':') + 1);
        if (family_settings.find(family) == family_settings.end())
            env::warn("warning: ignoring OP_STRATEGY item '" + std::string{item} +
                      "', which names no strategy family\n");
    }
}

void missing_family(std::string_view name, std::string_view loop, std::string_view backend) {
    std::fprintf(stderr, "error: %.*s registers family %.*s, which the %.*s backend lacks\n",
                 static_cast<int>(loop.size()), loop.data(), static_cast<int>(name.size()),
                 name.data(), static_cast<int>(backend.size()), backend.data());
    std::exit(1);
}

Ladder::Ladder(std::string_view loop_name, bool explore)
    : m_name{loop_name}, m_explore{explore}, m_forced_names{forced_for(loop_name)} {}

void Ladder::add(std::string_view family, const Make& make) {
    auto found = family_settings.find(family);
    if (found == family_settings.end()) {
        std::fprintf(stderr, "error: %s registers strategy family %s, which no backend declared\n",
                     m_name.c_str(), std::string{family}.c_str());
        std::exit(1);
    }
    const auto& settings = found->second;
    bool enabled = settings.enabled.value_or(settings.enabled_by_default);
    bool off = !enabled && settings.enabled.has_value();
    const auto& options = jit::options();

    // The family's other points, when the ladder explores, ahead of its point
    // of first values.
    auto all = points(family, settings.axes, settings.values);
    if (!m_explore)
        all.resize(1);
    std::rotate(all.begin(), all.begin() + 1, all.end());
    for (std::size_t i = 0; i < all.size(); ++i) {
        auto& strategy = *m_strategies.emplace_back(make(all[i]));
        for (const auto& other : m_strategies)
            if (other.get() != &strategy && other->name() == strategy.name()) {
                std::fprintf(stderr, "error: strategy %s registered twice (in %s)\n",
                             std::string{strategy.name()}.c_str(), m_name.c_str());
                std::exit(1);
            }

        bool member = i + 1 == all.size();
        auto add = [&](KernelKind kind, std::string_view refused = {}) {
            std::string name{strategy.name()};
            std::string short_name = label(family, all[i], all.back());
            if (kind == KernelKind::jit) {
                name += ";kernel=jit";
                short_name += " jit";
            }
            m_candidates.push_back({&strategy, kind, std::move(name), std::move(short_name),
                                    std::string{family}, settings.rank, member,
                                    member && enabled, off, refused});
        };
        auto offline = strategy.estimate(KernelKind::offline);
        auto jit = options.enabled() ? strategy.estimate(KernelKind::jit) : Estimate{};
        bool jit_first = jit.available &&
                         (!offline.available || options.force ||
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
    }
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
