#pragma once

// A loop's candidates, its strategies each with one of their kernels, in
// ladder order, and the selection of one per call.

#include <op2/rt/loop/call.h>
#include <op2/rt/loop/family.h>
#include <op2/rt/loop/strategy.h>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::loop {

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
// strategy for the point of every axis's first value, the one the ladder
// runs and OP_STRATEGY forces, and, in a ladder that explores, one for each
// other point of its axes, which only the tuner tries.  OP_<FAMILY>=on/off,
// read when the family is declared, wins over the default, and off turns all
// of the family off; OP_STRATEGY, read by init, is <family> or
// <loop>:<family> (a comma-separated list of either), a loop's own item
// winning over a family for every loop.
class Ladder {
private:
    std::string m_name;
    // Whether the families add their other points, for the tuner.
    bool m_explore;
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
    bool of_forced_family(const Candidate& candidate) const;
    void warn_forced(std::string_view reason);
    void report(const Prepared& selection,
                std::span<const std::string_view> reasons);

public:
    using Make = std::function<std::unique_ptr<Strategy>(const Point& point)>;

    Ladder(std::string_view loop_name, bool explore);
    Ladder(const Ladder&) = delete;

    const std::string& name() const { return m_name; }

    // A declared family: make builds its strategy for each point of its
    // axes, and the strategy's candidates follow from its estimates.
    void add(std::string_view family, const Make& make);

    // Each family the loop registers, made for executor as backend offers it;
    // a family the backend lacks stops the run.
    template<typename BackendExecutor, typename AbiFamily>
    void add_families(std::span<const AbiFamily> registered,
                      std::span<const OfferedFamily<BackendExecutor, AbiFamily>> offered,
                      BackendExecutor& executor, std::string_view backend) {
        for (const auto& family : registered) {
            auto found =
                std::ranges::find(offered, std::string_view{family.name}, &Family::name);
            if (found == offered.end())
                missing_family(family.name, m_name, backend);
            add(family.name, [&](const Point& point) {
                return found->make(executor, point, family);
            });
        }
    }

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
};

} // namespace op::rt::loop
