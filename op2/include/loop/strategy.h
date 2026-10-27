#pragma once

// A par_loop's execution strategies.  A strategy is a self-contained way of
// executing one loop; the ladder asks every strategy the same two questions
// and picks one per call.  Nothing here knows about kernels, devices or any
// one strategy family.

#include <op_lib_core.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::loop {

// Read the run-wide variables: OP_STRATEGY, the tuner's OP_AUTOTUNE* and the
// JIT's OP_JIT_*, and forget the families declared.  op_init calls it before
// the backend declares its families; a later call reads them again.
void init();

// --- One call -----------------------------------------------------------------

// One call, as a view: the set and the arguments.  Its configuration, the
// shape its key identifies, is what plans and the tuner depend on.
class Call {
private:
    mutable std::uint64_t m_key = 0;
    mutable bool m_keyed = false;

public:
    op_set set;
    std::span<const op_arg> args;
    // The elements the call runs, as op_mpi_halo_exchanges returned it: the
    // exec halo too when the exchange decided the call runs it.
    int run_size;

    Call(op_set set_, std::span<const op_arg> args_, int run_size_)
        : set{set_}, args{args_}, run_size{run_size_} {}

    // Any active global argument reduced across the set.
    bool global_reduction() const;

    // The set index, the run size and, per argument, its activity, kind, the
    // first argument on its dat, its map index, map component, access and
    // dimension: which arguments share a dat, not which dat, so a temporary
    // dat declared again keeps its plans.
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
    // Ending at run_size, which takes in the exec halo when the call runs it.
    static Sections indirect(op_set set, bool separate_owned, int run_size);
    // A call's sections: direct, or indirect with the owned elements apart
    // when it reduces a global.
    static Sections of(const Call& call, bool direct);

    int size() const { return m_count; }
    const Section& operator[](int index) const {
        return m_sections[static_cast<std::size_t>(index)];
    }
    std::span<const Section> span() const {
        return {m_sections.data(), static_cast<std::size_t>(m_count)};
    }

    // The section the halo wait comes before, in [0, size()], size() for
    // after the last; the section the globals are processed after.
    int wait_index() const { return m_wait_before; }
    int globals_index() const { return m_globals_after; }
    int set_stride() const { return m_set_stride; }
};

// What a strategy's run is given: the arguments, and whether to time the
// launches.
struct Launch {
    op_arg *args;
    int nargs;
    bool timed;
};

// A global's element type, from its C or Fortran spelling.
enum class ScalarType {
    f64,
    f32,
    i32,
    logical,
    unknown,
};

ScalarType scalar_type(const char *type);

// Reduce the call's INC, MIN and MAX globals across the ranks once this
// rank's elements are combined into them; an unsupported type stops the run,
// naming loop.
void reduce_mpi_globals(op_arg *args, int nargs, const char *loop);

// --- Strategies ---------------------------------------------------------------

enum class Status {
    ready,        // a kernel is ready for the configuration
    pending,      // its kernel is still being built
    unavailable,  // it does not run the configuration: no kernel, or a plan
                  // the configuration does not suit (the reason says which)
};

class Strategy;

// How soon a prepare needs its kernel: when the backend's policy gets to
// it, started now, or built before prepare returns.
enum class Urgency {
    later,
    now,
    wait,
};

// The kinds of kernel a strategy may run: the one compiled with the
// application, or one specialised at run time.
enum class KernelKind {
    offline,
    jit,
};

// A backend's estimate of one kind of a strategy's kernel for a
// configuration: whether it can run, and its expected time in the backend's
// own units, comparable only with the backend's other estimates for the
// configuration; none when the backend cannot say.
struct Estimate {
    bool available = false;
    std::optional<float> time;
};

// A strategy's answer for one configuration.  Opaque to everything but the
// strategy that produced it; carries no ownership.
struct Prepared {
    // The strategy that prepared it, which runs it.
    Strategy *strategy = nullptr;
    Status status = Status::unavailable;
    // A fixed identifier when not ready: "low_compression", "compiling",
    // "kernel_unavailable", ...
    std::string_view reason;
    // The strategy's cached entry and kernel for the configuration.
    const void *entry = nullptr;
    void *kernel = nullptr;
};

class Strategy {
public:
    virtual ~Strategy() = default;

    // The family and each axis's value, "chunk_atomics=on;block_size=64;
    // chunk_size=128;singles=on"; the report column.
    virtual std::string_view name() const = 0;

    // Estimate its kernel of kind for call: an offline one at a nominal 1
    // and no specialised one, unless the backend says otherwise.
    virtual Estimate estimate(const Call& call, KernelKind kind) const {
        (void)call;
        if (kind == KernelKind::offline)
            return {true, 1.0f};
        return {};
    }

    // Resolve call: build or find its configuration's plan, and find or
    // request its kernel, as urgency asks.  Blocks only for Urgency::wait, or when the
    // backend builds synchronously.  The kernel is a specialised one when the
    // JIT is on and one can run, and the offline one cannot, OP_JIT_FORCE is
    // set, or the estimates expect it faster; else the offline one.  A tie
    // or a missing estimate keeps the offline kernel, as a specialised one
    // costs a build.
    Prepared prepare(const Call& call, Urgency urgency);

    // prepare with a kernel of kind.
    virtual Prepared prepare_kind(const Call& call, Urgency urgency,
                                  KernelKind kind) = 0;

    // Run what prepare returned; microseconds when the launch is timed.  A
    // run always follows this strategy's prepare for the same call, so the
    // strategy may keep per-call state from it; other strategies of the loop
    // may be prepared in between, so state they share must not carry it.
    virtual std::optional<float> run(const Prepared& prepared,
                                     const Launch& launch) = 0;

    // Figures for the report row: compression, colours, the kernel kind.
    // Called once per trial, when it completes.
    virtual std::string statistics(const Prepared& prepared) const {
        (void)prepared;
        return {};
    }
};

// --- Variants -----------------------------------------------------------------

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

// A loop's strategies in ladder order, their families' as declared,
// whatever order the loop adds them in.  The last one is the floor: it is
// selected when no other is ready, enabled or not, whatever its own status.
// A family adds a strategy per point of its axes; the point of every axis's
// first value is the one the ladder runs and OP_STRATEGY forces, the others
// only the tuner tries.  OP_<FAMILY>=on/off, read when the family is
// declared, wins over the default, and off turns all of the family off;
// OP_STRATEGY, read by init, is <family> or <loop>:<family> (a
// comma-separated list of either), a loop's own item winning over a family
// for every loop.
class Ladder {
private:
    struct Entry {
        std::unique_ptr<Strategy> strategy;
        std::string family;
        // The family's point of first values, which the ladder may run.
        bool runs;
        bool enabled;
        // Turned off by its variable: never run and never explored.
        bool off;
        // Its family's place in ladder order.
        std::size_t rank;
    };

    std::string m_name;
    // OP_STRATEGY's families for this loop, most specific first, and the
    // rank of the one forced.
    std::vector<std::string> m_forced_names;
    std::size_t m_forced_rank = SIZE_MAX;
    std::vector<Entry> m_entries;
    std::optional<std::size_t> m_forced;
    // Why select passed over each strategy, sized as they are added so a
    // select allocates nothing.
    std::vector<std::string_view> m_reasons;
    std::vector<std::string> m_reported;
    // Why the forced strategy was passed over, each warned about once.
    std::vector<std::string> m_warned;

    void add_entry(std::unique_ptr<Strategy> strategy, std::string_view family,
                   bool runs, bool enabled, bool off, std::size_t rank);
    void warn_forced(std::string_view reason);
    void report(const Prepared& selection,
                std::span<const std::string_view> reasons);

public:
    using Make = std::function<std::unique_ptr<Strategy>(const Point& point)>;

    explicit Ladder(std::string_view loop_name);
    Ladder(const Ladder&) = delete;

    const std::string& name() const { return m_name; }

    // A declared family: make builds its strategy for each point of its
    // axes, the tuner's ahead of the one that runs, so a family's last
    // strategy is the one that runs.
    void add(std::string_view family, const Make& make);

    // First enabled strategy, from the forced one if any, whose prepare is
    // ready; else the floor, waited for, so the selection is always ready: a
    // floor that cannot run stops the program.  The forced strategy is
    // waited for too, and each reason it is still passed over is warned
    // about once.  At diagnostic verbosity each distinct outcome (what ran,
    // and why each earlier strategy did not) is printed once per loop.
    Prepared select(const Call& call);

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

// Owners of device state register a release callback, which finalise
// (loop/autotune.h) runs at op_exit, before the device is torn down.
void register_release(void *owner, void (*release)(void *owner));
void unregister_release(void *owner);
void release_all();

} // namespace op::loop
