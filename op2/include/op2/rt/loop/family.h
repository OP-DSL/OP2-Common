#pragma once

// The strategy families a backend declares, the axes their strategies vary
// along, and OP_STRATEGY, which forces a family.  Defined in
// src/rt/loop/op_loop_ladder.cpp, whose ladders take the declared families'
// settings.

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::loop {

// Read OP_STRATEGY and forget the families declared; op_init calls it, through
// op::rt::init, before the backend declares its families, and a later call
// reads it again.
void init();

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
};

// A strategy's place on its family's axes: its name and a value per axis,
// under the axis's key.
struct Point {
    std::string name;
    std::vector<std::string_view> keys;
    std::vector<int> values;

    int value(std::string_view key) const;
};

// A strategy family a backend offers, under the name the generated code
// registers it by, its axes, and whether the ladder runs it by default.
struct Family {
    std::string_view name;
    std::vector<Axis> axes;
    bool enabled = true;
};

// Read each family's variables, OP_<FAMILY> and its axes'; the backend
// declares its families in ladder order as op_init runs, and a ladder takes
// their settings and order from here.
void declare_families(std::span<const Family> families);

// Stop the run: loop registers a family the backend lacks.
[[noreturn]] void missing_family(std::string_view name, std::string_view loop,
                                 std::string_view backend);

// The family named name among those a backend offers, which loop registers.
template<typename Offered>
const Offered& offered_family(std::span<const Offered> offered, std::string_view name,
                              std::string_view loop, std::string_view backend) {
    for (const auto& family : offered)
        if (family.name == name)
            return family;
    missing_family(name, loop, backend);
}

} // namespace op::rt::loop
