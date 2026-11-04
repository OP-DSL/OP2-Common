#pragma once

// A par_loop's execution strategies.  A strategy is a self-contained way of
// executing one loop, with an offline kernel, a specialised one, or both; a
// candidate is a strategy with one of its kernels (op2/rt/loop/ladder.h).
// Nothing here knows about devices or any one strategy family.

#include <op2/rt/loop/call.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace op::rt::loop {

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

    static Prepared ready(const void *entry = nullptr) {
        Prepared prepared;
        prepared.status = Status::ready;
        prepared.entry = entry;
        return prepared;
    }

    static Prepared unavailable(std::string_view reason, const void *entry = nullptr) {
        Prepared prepared;
        prepared.reason = reason;
        prepared.entry = entry;
        return prepared;
    }
};

// A cache's builds by where they stand: built, failed, under way, and keys
// asked for whose builds have not started.
struct BuildCounts {
    std::size_t built = 0;
    std::size_t failed = 0;
    std::size_t building = 0;
    std::size_t waiting = 0;

    BuildCounts& operator+=(const BuildCounts& other) {
        built += other.built;
        failed += other.failed;
        building += other.building;
        waiting += other.waiting;
        return *this;
    }
};

// A backend's estimate of one kind of a strategy's kernel before anything is
// timed: whether the strategy has it, why not ("kernel_unavailable"), and
// its cost, dimensionless and comparable only with the loop's other
// estimates, none when the backend cannot say.  The ladder orders a
// strategy's kernels by it, and the tuner's timings overrule it once a
// configuration is explored.
struct Estimate {
    bool available = false;
    std::string_view reason{};
    std::optional<float> cost{};
};

class Strategy {
public:
    virtual ~Strategy() = default;

    // The family and each axis's value, "chunk_atomics=on;block_size=64;
    // singles=on"; the report column.
    virtual std::string_view name() const = 0;

    // Its kernel of kind as its backend's heuristics estimate it before any
    // call: an offline one at a nominal 1 and no specialised one, unless the
    // backend says otherwise.  Fixed once the strategy is built: a kind it
    // does not have is never prepared.
    virtual Estimate estimate(KernelKind kind) const {
        return kind == KernelKind::offline ? Estimate{.available = true, .cost = 1.0f}
                                           : Estimate{};
    }

    // Resolve call for its kernel of kind: build or find the configuration's
    // plan, and find or request the kernel, as urgency asks.  Blocks only for
    // Urgency::wait, or when the backend builds synchronously.
    virtual Prepared prepare(const Call& call, Urgency urgency, KernelKind kind) = 0;

    // Once its candidate is selected for the call, before the run: set up
    // what the run needs of the call's globals.
    virtual void begin_call(const Prepared& prepared, const Call& call) {
        (void)prepared;
        (void)call;
    }

    // Run what prepare returned for call; microseconds when timed.  What the
    // run needs comes from the call and the prepared entry, and what
    // strategies share from their executor, as begin_call set it up.
    virtual std::optional<float> run(const Prepared& prepared, const Call& call,
                                     bool timed) = 0;

    // Figures for the report row: compression, colours, the kernel kind.
    // Called once per trial, when it completes.
    virtual std::string statistics(const Prepared& prepared) const {
        (void)prepared;
        return {};
    }

    // Drop its plans for the configuration key (Call::key): the tuner has
    // explored it, and nothing will run this strategy on it again.  A later
    // prepare plans it anew.
    virtual void forget(std::uint64_t key) { (void)key; }

    // Its specialised kernels' builds, by where they stand; none without a
    // JIT.
    virtual BuildCounts builds() const { return {}; }
};

} // namespace op::rt::loop
