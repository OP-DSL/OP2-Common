#include <op2/rt/loop/family.h>

#include <op2/env.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace op::rt::loop {

std::string Axis::item(int value) const {
    return std::string{key} + "=" +
           (binary ? (value != 0 ? "on" : "off") : std::to_string(value));
}

int Point::value(std::string_view key) const {
    auto found = std::find(keys.begin(), keys.end(), key);
    assert(found != keys.end());
    return values[static_cast<std::size_t>(found - keys.begin())];
}

namespace {

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
        } else if (auto parsed = env::number_value(item, 1)) {
            value = *parsed;
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
        auto chosen = axis_values("OP_" + env::upper(family) + "_" + variable, axis);
        if (chosen.empty())
            chosen = axis_values("OP_" + variable, axis);
        if (chosen.empty())
            chosen = axis.defaults;
        assert(!chosen.empty());
        values.push_back(std::move(chosen));
    }
    return values;
}

} // namespace

std::vector<Point> Families::Settings::points(std::string_view family) const {
    std::vector<Point> out;
    std::vector<std::size_t> index(axes.size(), 0);
    while (true) {
        Point point{std::string{family} + "=on", {}, {}};
        for (std::size_t i = 0; i < axes.size(); ++i) {
            int value = values[i][index[i]];
            point.keys.push_back(axes[i].key);
            point.values.push_back(value);
            point.name += ";" + axes[i].item(value);
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

Families::Families() { env::parse_list("OP_STRATEGY", m_forced); }

void Families::declare(std::span<const Family> families) {
    for (const auto& family : families) {
        Settings settings;
        settings.axes = family.axes;
        settings.enabled_by_default = family.enabled;
        env::parse_bool(("OP_" + env::upper(family.name)).c_str(), settings.enabled);
        settings.values = family_values(family.name, family.axes);
        settings.rank = m_declared++;
        m_settings.insert_or_assign(std::string{family.name}, std::move(settings));
    }
    for (std::string_view item : m_forced) {
        auto family = item.substr(item.find(':') + 1);
        if (m_settings.find(family) == m_settings.end())
            env::warn("warning: ignoring OP_STRATEGY item '" + std::string{item} +
                      "', which names no strategy family\n");
    }
}

const Families::Settings *Families::find(std::string_view family) const {
    auto found = m_settings.find(family);
    return found == m_settings.end() ? nullptr : &found->second;
}

std::vector<std::string> Families::forced_for(std::string_view loop) const {
    std::vector<std::string> specific;
    std::vector<std::string> every;
    for (std::string_view item : m_forced)
        if (auto scoped = env::scoped_item(item, loop))
            (scoped->own ? specific : every).emplace_back(scoped->name);
    specific.insert(specific.end(), every.begin(), every.end());
    return specific;
}

void missing_family(std::string_view name, std::string_view loop, std::string_view backend) {
    std::fprintf(stderr, "error: %.*s registers family %.*s, which the %.*s backend lacks\n",
                 static_cast<int>(loop.size()), loop.data(), static_cast<int>(name.size()),
                 name.data(), static_cast<int>(backend.size()), backend.data());
    std::exit(1);
}

} // namespace op::rt::loop
