#pragma once

// The strategy families a backend declares, the axes their strategies vary
// along, and OP_STRATEGY, which forces a family.  Defined in
// src/rt/loop/op_loop_family.cpp; ladders take the declared families'
// settings.

#include <op2/rt/loop/strategy.h>

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::loop {

// One way a family's strategies vary, such as their block size.  Its values
// are OP_<FAMILY>_<VARIABLE>, else OP_<VARIABLE>, else the defaults: one
// value or a comma-separated list, the first for the strategy that runs and
// the rest for the tuner.  A switch takes on or off, any other axis positive
// numbers; a strategy's name gives each value under the axis's key.
struct Axis {
    std::string_view variable;
    std::string_view key;
    bool binary = false;
    std::vector<int> defaults;

    // value as a strategy's name gives it: "block_size=64", "singles=off".
    std::string item(int value) const;
};

// A strategy's place on its family's axes: its name and a value per axis,
// under the axis's key.
struct Point {
    std::string name;
    std::vector<std::string_view> keys;
    std::vector<int> values;

    int value(std::string_view key) const;
    bool has(std::string_view key) const;
};

// A strategy family a backend offers, under the name the generated code
// registers it by, its axes, and whether the ladder runs it by default.
struct Family {
    std::string_view name;
    std::vector<Axis> axes;
    bool enabled = true;
};

// A family as a backend offers it: as declared, and the strategy it makes at
// each point for a loop's BackendExecutor, under the family's settings in
// the loop's descriptor, an AbiFamily.
template<typename BackendExecutor, typename AbiFamily>
struct OfferedFamily : Family {
    std::unique_ptr<Strategy> (*make)(BackendExecutor& executor, const Point& point,
                                      const AbiFamily& family) = nullptr;

    // The family name of strategies S, which the ladder runs unless enabled
    // says otherwise.
    template<typename S>
    static OfferedFamily of(std::string_view name, bool enabled = true) {
        return {{name, S::axes(), enabled},
                [](BackendExecutor& executor, const Point& point,
                   const AbiFamily& family) -> std::unique_ptr<Strategy> {
                    return std::make_unique<S>(executor, point, family);
                }};
    }
};

// The families the backend declared, in ladder order, each with the
// settings its variables give it, and OP_STRATEGY's items, which force a
// family.  The run's context holds them (op2/rt/context.h).
class Families {
public:
    // A declared family's axes, default, OP_<FAMILY>, each axis's values, and
    // its place in ladder order.
    struct Settings {
        std::vector<Axis> axes;
        bool enabled_by_default;
        std::optional<bool> enabled;
        std::vector<std::vector<int>> values;
        std::size_t rank;

        // Every point of the values, the first values' first, named
        // "<family>=on;<key>=<value>;...", a switch's value on or off.
        std::vector<Point> points(std::string_view family) const;
    };

    // Read OP_STRATEGY.
    Families();

    // Read each family's variables, OP_<FAMILY> and its axes'; the backend
    // declares its families in ladder order as op_init runs, and a ladder
    // takes their settings and order from here.
    void declare(std::span<const Family> families);

    template<typename BackendExecutor, typename AbiFamily>
    void declare(std::span<const OfferedFamily<BackendExecutor, AbiFamily>> offered) {
        std::vector<Family> families(offered.begin(), offered.end());
        declare(families);
    }

    // A declared family's settings; none if no backend declared it.
    const Settings *find(std::string_view family) const;

    // The families forced on loop, most specific first: "<loop>:<family>"
    // forces that loop, a bare family every loop that has it.
    std::vector<std::string> forced_for(std::string_view loop) const;

private:
    std::vector<std::string> m_forced;
    std::map<std::string, Settings, std::less<>> m_settings;
    std::size_t m_declared = 0;
};

// Stop the run: loop registers a family the backend lacks.
[[noreturn]] void missing_family(std::string_view name, std::string_view loop,
                                 std::string_view backend);

} // namespace op::rt::loop
