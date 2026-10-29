#pragma once

// A par_loop's execution strategies.  A strategy is a self-contained way of
// executing one loop, with an offline kernel, a specialised one, or both; a
// candidate is a strategy with one of its kernels (op2/rt/loop/ladder.h).
// Nothing here knows about devices or any one strategy family.

#include <op2/rt/abi/descriptor.h>
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

// What a backend knows of one kind of a strategy's kernel before it runs:
// whether the strategy has it, why not ("kernel_unavailable"), and its
// compiler's figures once known.
struct KernelFacts {
    bool available = false;
    std::string_view reason;
    std::optional<abi::KernelFigures> figures;
};

// A strategy's facts, which the heuristics weigh.
struct Facts {
    KernelFacts offline;
    KernelFacts jit;
};

class Strategy {
public:
    virtual ~Strategy() = default;

    // The family and each axis's value, "chunk_atomics=on;block_size=64;
    // singles=on"; the report column.
    virtual std::string_view name() const = 0;

    // What it knows of its kernels before any call, which the heuristics
    // (op2/rt/tune/heuristics.h) estimate them from: an offline one with no
    // figures and no specialised one, unless the backend says otherwise.
    // Fixed once the strategy is built: a kind it does not have is never
    // prepared.
    virtual Facts facts() const { return {{true, {}, {}}, {}}; }

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
                                     const RunArgs& run) = 0;

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

    // At op_exit, while the device is still up: free its device state.
    virtual void release() {}
};

} // namespace op::rt::loop
