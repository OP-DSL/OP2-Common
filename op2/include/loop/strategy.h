#pragma once

// A par_loop's execution strategies.  A strategy is a self-contained way of
// executing one loop; the ladder asks every strategy the same two questions
// and picks one per call.  Nothing here knows about kernels, devices or any
// one strategy family.

#include <op_lib_core.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::loop {

// --- Environment variables ---------------------------------------------------

namespace env {

std::string lower(std::string_view text);

// Parse a 0/1 variable into value, warning about anything else; whether the
// variable was set.
bool parse_bool(const char *variable, bool& value);

// Parse a non-negative number into value, warning about anything else;
// whether the variable was set.
bool parse_number(const char *variable, double& value);
bool parse_number(const char *variable, int& value);

} // namespace env

// --- One call -----------------------------------------------------------------

// One call, as a view: the set and the arguments.  The key is the call's
// shape, which is what plans and the tuner depend on.
class Configuration {
private:
    mutable std::uint64_t m_key = 0;
    mutable bool m_keyed = false;

public:
    op_set set;
    std::span<const op_arg> args;

    Configuration(op_set set_, std::span<const op_arg> args_)
        : set{set_}, args{args_} {}

    // Any active global argument reduced across the set.
    bool global_reduction() const;

    // The set index and, per argument, its activity, kind, dat and map
    // indices, map component, access and dimension.
    std::uint64_t key() const;

    // "edges (3.1M) via edge_nodes, reduction"; for reports only.
    std::string describe() const;
};

// The element ranges a loop runs, in order, with where its halo wait and
// global processing fall among them.  A direct loop reads no halo: its
// exchange only brings the dats to this device, so it waits before its one
// section.  An indirect loop runs its core elements, then, once the halo has
// arrived, the rest; the owned elements form a section of their own when a
// global reduction has to be processed after them.
struct Section {
    int begin = 0;
    int end = 0;

    int size() const { return end - begin; }
};

// The stride of an indirect loop's per-element arrays: the set's owned and
// exec-halo elements, rounded up to a warp.
inline int indirect_set_stride(op_set set) {
    return (set->size + set->exec_size + 31) & ~31;
}

class Sections {
private:
    std::array<Section, 3> m_sections{};
    int m_count = 0;
    int m_wait_before = 0;
    int m_globals_after = 0;
    int m_set_stride = 0;

public:
    static Sections direct(op_set set);
    static Sections indirect(op_set set, bool separate_owned);

    int size() const { return m_count; }
    const Section& operator[](int index) const {
        return m_sections[static_cast<std::size_t>(index)];
    }
    std::span<const Section> span() const {
        return {m_sections.data(), static_cast<std::size_t>(m_count)};
    }

    // Exactly one index in [0, size()] answers true; size() means after the
    // last section.
    bool wait_before(int index) const { return index == m_wait_before; }
    bool process_globals_after(int index) const {
        return index == m_globals_after;
    }
    int set_stride() const { return m_set_stride; }
};

// What a strategy's run is given: the arguments, and whether to time the
// launches.
struct Launch {
    op_arg *args;
    int nargs;
    bool timed;
};

// --- Strategies ---------------------------------------------------------------

enum class Status {
    ready,        // a kernel is ready for the configuration
    pending,      // its kernel is still being built
    unavailable,  // it never has a kernel for it
    rejected,     // the configuration does not suit it
};

class Strategy;

// A strategy's answer for one configuration.  Opaque to everything but the
// strategy that produced it; carries no ownership.
struct Prepared {
    Strategy *strategy = nullptr;
    Status status = Status::unavailable;
    // A fixed identifier when not ready: "disabled", "low_compression",
    // "compiling", ...
    std::string_view reason;
    // The strategy's cached entry and kernel for the configuration.
    const void *entry = nullptr;
    void *kernel = nullptr;
};

class Strategy {
public:
    virtual ~Strategy() = default;

    // "atomics", or a variant "atomics:256".  The part before the colon is
    // the family, which the variable OP_<FAMILY>=0/1 enables; the full name
    // is what OP_STRATEGY forces and the report column.
    virtual std::string_view name() const = 0;

    // Resolve the configuration: build or find its plan, find or request its
    // kernel, at once when now.  Never blocks unless the backend builds
    // synchronously and now is set.
    virtual Prepared prepare(const Configuration& configuration, bool now) = 0;

    // Run what prepare returned; microseconds when the launch is timed.
    virtual std::optional<float> run(const Prepared& prepared,
                                     const Launch& launch) = 0;

    // Figures for the report row: compression, colours, the kernel kind.
    // Called once per trial, when it completes.
    virtual std::string statistics(const Prepared& prepared) const {
        (void)prepared;
        return {};
    }

    // The family: the name up to its colon.
    std::string_view family() const;
};

enum class Default {
    enabled,
    disabled,
};

struct Selection {
    Strategy *strategy = nullptr;
    Prepared prepared;
};

// A loop's strategies in ladder order.  The last one is the floor: it is
// selected when no other is ready, enabled or not, whatever its own status.
// Variables are read at registration: OP_<FAMILY>=0/1 over the default, and
// OP_STRATEGY=<name> or OP_STRATEGY=<loop>:<name> (a comma-separated list of
// either) forcing one.
class Ladder {
private:
    struct Entry {
        std::unique_ptr<Strategy> strategy;
        bool enabled;
        // Turned off by its variable: never run and never explored.
        bool off;
    };

    std::string m_name;
    std::optional<std::string> m_forced_name;
    std::vector<Entry> m_entries;
    std::optional<std::size_t> m_forced;
    std::vector<std::string> m_reported;

    void report(const Selection& selection,
                std::span<const std::string_view> reasons);

public:
    explicit Ladder(std::string_view loop_name);
    Ladder(const Ladder&) = delete;

    const std::string& name() const { return m_name; }

    void add(std::unique_ptr<Strategy> strategy, Default default_setting);

    // First enabled strategy, from the forced one if any, whose prepare is
    // ready; else the floor's prepared value, which may not be ready.  At
    // diagnostic verbosity each distinct outcome (what ran, and why each
    // earlier strategy did not) is printed once per loop.
    Selection select(const Configuration& configuration);

    std::size_t size() const { return m_entries.size(); }
    Strategy& operator[](std::size_t index) { return *m_entries[index].strategy; }
    const Strategy& operator[](std::size_t index) const {
        return *m_entries[index].strategy;
    }
    bool enabled(std::size_t index) const { return m_entries[index].enabled; }
    // Not turned off by its variable: the tuner explores these.
    bool explorable(std::size_t index) const { return !m_entries[index].off; }
    std::optional<std::size_t> forced() const { return m_forced; }
};

// --- Teardown -----------------------------------------------------------------

// Owners of device state register a release callback, which op_exit runs
// before tearing the device down.
void register_release(void *owner, void (*release)(void *owner));
void unregister_release(void *owner);
void release_all();

} // namespace op::loop
