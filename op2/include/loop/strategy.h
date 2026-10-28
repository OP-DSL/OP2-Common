#pragma once

// A par_loop's execution strategies.  A strategy is a self-contained way of
// executing one loop, with an offline kernel, a specialised one, or both; a
// candidate is a strategy with one of its kernels, and the ladder picks a
// candidate per call.  Nothing here knows about devices or any one strategy
// family.

#include <op_lib_core.h>

#include <array>
#include <cstdint>
#include <deque>
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
    ready,        // its kernel is ready for the configuration
    pending,      // its kernel is still being built
    unavailable,  // it does not run the configuration: no kernel, or a plan
                  // the configuration does not suit (the reason says which)
};

class Strategy;
struct Candidate;

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

// A backend's estimate of one kind of a strategy's kernel: whether the
// strategy has it, and its expected time in the backend's own units,
// comparable only with the backend's other estimates for the loop, none when
// the backend cannot say; when it has none, why ("kernel_unavailable").
struct Estimate {
    bool available = false;
    std::optional<float> time;
    std::string_view reason{};
};

// A candidate's answer for one configuration.  Opaque to everything but the
// strategy that produced it; carries no ownership.
struct Prepared {
    // The candidate prepared, whose strategy runs it; set by
    // Candidate::prepare.
    const Candidate *candidate = nullptr;
    Status status = Status::unavailable;
    // A fixed identifier when not ready: "low_compression", "compiling",
    // "kernel_unavailable", ...
    std::string_view reason;
    // The strategy's cached entry for the configuration, and its specialised
    // kernel; none for an offline one.
    const void *entry = nullptr;
    void *kernel = nullptr;
};

class Strategy {
public:
    virtual ~Strategy() = default;

    // The family and each axis's value, "chunk_atomics=on;block_size=64;
    // singles=on"; the report column.
    virtual std::string_view name() const = 0;

    // Its kernel of kind: an offline one at a nominal 1 and no specialised
    // one, unless the backend says otherwise.  Fixed once the strategy is
    // built, and independent of any call: a kind it does not have is never
    // prepared.
    virtual Estimate estimate(KernelKind kind) const {
        if (kind == KernelKind::offline)
            return {true, 1.0f};
        return {};
    }

    // Resolve call for its kernel of kind: build or find the configuration's
    // plan, and find or request the kernel, as urgency asks.  Blocks only for
    // Urgency::wait, or when the backend builds synchronously.
    virtual Prepared prepare(const Call& call, Urgency urgency, KernelKind kind) = 0;

    // Run what prepare returned; microseconds when the launch is timed.  A
    // run always follows this strategy's prepare for the same call, so the
    // strategy may keep per-call state from it.  Its other kernel's prepare
    // and other strategies' may come in between, so that state must not
    // depend on the kind, and state strategies share must not carry it.
    virtual std::optional<float> run(const Prepared& prepared,
                                     const Launch& launch) = 0;

    // Figures for the report row: compression, colours, the kernel kind.
    // Called once per trial, when it completes.
    virtual std::string statistics(const Prepared& prepared) const {
        (void)prepared;
        return {};
    }

    // Drop its plans for the configuration whose key is call: the tuner has
    // explored it, and nothing will run this strategy on it again.  A later
    // prepare plans it anew.
    virtual void forget(std::uint64_t call) { (void)call; }
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
    // Or otherwise, when its family has no such axis.
    int value(std::string_view key, int otherwise) const;
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

// A strategy with one of its kernels: what the ladder runs and the tuner
// times.  A strategy has a candidate per kernel it has, and one with neither
// none, unless it is the ladder's point of its family: that keeps an offline
// candidate, refused with the estimates' reason.
struct Candidate {
    Strategy *strategy;
    KernelKind kind;
    // The strategy's name, ";kernel=jit" added for a specialised kernel; and
    // for summaries the family, the axis values where its point differs from
    // the ladder's, and "jit" for a specialised kernel.
    std::string name;
    std::string label;
    std::string family;
    // Its family's place in ladder order.
    std::size_t rank;
    // Of its family's point of first values, which the ladder may run.
    bool member;
    // A member of a family on by default or by its variable.
    bool enabled;
    // Turned off by its variable: never run or explored.
    bool off;
    // Why its strategy has no kernel at all; such a candidate is
    // unavailable, its strategy never asked.
    std::string_view refused;

    Prepared prepare(const Call& call, Urgency urgency) const;
};

// A loop's candidates in ladder order: its families' as declared, whatever
// order the loop adds them in, a family's other points ahead of the one the
// ladder runs, and a strategy's kernels in the order of their estimates (the
// specialised one first when it is expected faster, or under
// OP_JIT_FORCE).  The last candidate is the floor: it is selected when no
// other is ready, enabled or not, whatever its own status.  A family adds a
// strategy per point of its axes; the point of every axis's first value is
// the one the ladder runs and OP_STRATEGY forces, the others only the tuner
// tries.  OP_<FAMILY>=on/off, read when the family is declared, wins over the
// default, and off turns all of the family off; OP_STRATEGY, read by init, is
// <family> or <loop>:<family> (a comma-separated list of either), a loop's
// own item winning over a family for every loop.
class Ladder {
private:
    std::string m_name;
    // OP_STRATEGY's families for this loop, most specific first.
    std::vector<std::string> m_forced_names;
    std::vector<std::unique_ptr<Strategy>> m_strategies;
    // In the order added, which references stay valid through; m_order holds
    // ladder order.
    std::deque<Candidate> m_candidates;
    std::vector<std::size_t> m_order;
    // The forced family's first and last member, in ladder order.
    std::optional<std::size_t> m_forced;
    std::size_t m_forced_last = 0;
    // Why select passed over each candidate, sized as they are added so a
    // select allocates nothing.
    std::vector<std::string_view> m_reasons;
    std::vector<std::string> m_reported;
    // Why the forced family was passed over, each warned about once.
    std::vector<std::string> m_warned;

    void order();
    void warn_forced(std::string_view reason);
    void report(const Prepared& selection,
                std::span<const std::string_view> reasons);

public:
    using Make = std::function<std::unique_ptr<Strategy>(const Point& point)>;

    explicit Ladder(std::string_view loop_name);
    Ladder(const Ladder&) = delete;

    const std::string& name() const { return m_name; }

    // A declared family: make builds its strategy for each point of its
    // axes, and the strategy's candidates follow from its estimates.
    void add(std::string_view family, const Make& make);

    // First enabled member, from the forced family if any, whose prepare is
    // ready; else the floor, waited for, so the selection is always ready: a
    // floor that cannot run stops the program.  The forced family's members
    // start their builds at once and its last is waited for, and each reason
    // the family is still passed over is warned about once.  At diagnostic
    // verbosity each distinct outcome (what ran, and why each earlier
    // candidate did not) is printed once per loop.
    Prepared select(const Call& call);

    const std::vector<std::unique_ptr<Strategy>>& strategies() const {
        return m_strategies;
    }

    // Whether select may run candidate: a member of an enabled or the forced
    // family, or the floor.
    bool runs(const Candidate& candidate) const;

    // The candidates in ladder order.
    std::size_t size() const { return m_order.size(); }
    const Candidate& operator[](std::size_t index) const {
        return m_candidates[m_order[index]];
    }
    std::optional<std::size_t> forced() const { return m_forced; }
};

} // namespace op::loop
