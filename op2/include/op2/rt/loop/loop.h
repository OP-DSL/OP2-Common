#pragma once

// A par_loop's loop, made from the generated code's descriptor on its first
// call: its JIT parameters, its backend's executor, its ladder and its
// tuner, and a call's steps, phases and profile scopes, from
// src/rt/loop/op_loop.cpp.  The backend's strategies prepare and combine the
// call's globals and time their phases.

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/jit/params.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/ladder.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/tune/autotune.h>
#include <op_lib_core.h>
#include <op_profile.h>

#include <memory>
#include <optional>
#include <span>
#include <string>

namespace op::rt::loop {

class Loop;

// Every loop made for a descriptor, which the reports read: the registry
// owns it until finalise.
void register_loop(std::unique_ptr<Loop> loop, abi::Loop& descriptor);

// At op_profile_output and op_exit: write the tuner's and the JIT's reports,
// OP_AUTOTUNE_REPORT and OP_JIT_REPORT, suffixed with the rank when there
// is more than one, and print their summaries of this rank's loops once, on
// rank 0.  A later call writes the files again, so applications that never
// call op_exit still get them.
void write_reports(int rank, int ranks);

// At op_exit, while the device is still up: write the reports, then destroy
// every loop, which frees its device state, and clear its descriptor, so a
// later call makes a new one.
void finalise(int rank, int ranks);

// Wait for the call's halo exchanges.
void wait_halos(const Call& call);

// Run a strategy's phases, phase(index) each, with the halo wait and, when
// the call writes globals, their combine() where phases puts them.
template<typename Phase, typename Combine>
void run_phases(const Call& call, const Phases& phases, Phase&& phase, Combine&& combine) {
    for (int index = 0; index < phases.count; ++index) {
        if (index == phases.wait_before)
            wait_halos(call);
        phase(index);
        if (index == phases.globals_after && call.writes_globals()) {
            op_profile_next("Process GBLs");
            combine();
            op_profile_next("Kernel");
        }
    }
    if (phases.count == phases.wait_before)
        wait_halos(call);
}

// What a loop's strategies share on its backend, such as a compiler or the
// threads' copies of the globals, and where the loop's data lives.  The loop
// holds it ahead of its ladder, so it is built before the strategies and
// destroyed after them.
class Executor {
public:
    virtual ~Executor() = default;

    // Add the strategies of the families the loop registers.
    virtual void add_strategies(Ladder& ladder) = 0;

    // Exchange the call's halos where the data lives, the host's unless the
    // executor says otherwise; the elements the call runs.
    virtual int exchange_halos(op_set set, std::span<op_arg> args) {
        return op_mpi_halo_exchanges(set, args, 1);
    }

    // Mark the halos of the dats the call wrote out of date, where they live.
    virtual void mark_dirty(std::span<op_arg> args) { op_mpi_set_dirtybit(args); }
};

class Loop final {
private:
    std::string m_name;
    std::string m_profile_target;
    std::string m_profile_variant;
    bool m_direct;
    jit::ParamSet m_params;
    std::unique_ptr<Executor> m_executor;
    Ladder m_ladder;
    std::optional<tune::Autotuner> m_tuner;

    // A call's decision: the parameters take its values, and the ladder
    // selects, or with a tuner the tuner chooses, asking the ladder when it
    // needs its selection.  Every call counts in the parameters' statistics,
    // whichever candidate runs, so a specialised kernel meets settled modes,
    // and the tuner waits for them.
    tune::Autotuner::Decision decide(const Call& call);

    // Run a decision, its time a sample for the tuner when it is timed.
    void run(const tune::Autotuner::Decision& decision, const Call& call);

public:
    // The loop of descriptor, which names it to the profiler, and its
    // executor make(params), which registers the loop's parameters in params.
    template<typename Make>
    Loop(const abi::Loop& descriptor, Make&& make)
        : m_name{descriptor.name}, m_profile_target{descriptor.target},
          m_profile_variant{descriptor.variant}, m_direct{descriptor.direct},
          m_params{m_name}, m_executor{make(m_params)},
          m_ladder{m_name, tune::autotune_options().report} {
        m_executor->add_strategies(m_ladder);
        if (tune::autotune_options().report)
            m_tuner.emplace(m_ladder);
    }

    Loop(const Loop&) = delete;

    const std::string& name() const { return m_name; }
    Ladder& ladder() { return m_ladder; }
    const Ladder& ladder() const { return m_ladder; }
    const jit::ParamSet& params() const { return m_params; }
    const tune::Autotuner *tuner() const { return m_tuner ? &*m_tuner : nullptr; }

    // Its specialised kernels' builds, by where they stand.
    BuildCounts builds() const {
        BuildCounts counts;
        for (const auto& strategy : m_ladder.strategies())
            counts += strategy->builds();
        return counts;
    }

    // Run one call: start the halo exchange, which decides what the call runs
    // and which the planning overlaps; decide; prepare the globals; run; and
    // reduce and mark the globals and dats.
    void invoke(op_set set, std::span<op_arg> args);
};

// The library's loop for a descriptor, made on its first call with a
// BackendExecutor from the descriptor and the loop's parameters: each
// backend's invoke runs it.  It lives until finalise, which clears the
// descriptor, so the descriptor must outlive it: the generated code's are
// static.
template<typename BackendExecutor, typename Descriptor>
Loop& runtime(Descriptor& descriptor) {
    if (descriptor.runtime == nullptr) {
        auto made = std::make_unique<Loop>(descriptor, [&](jit::ParamSet& params) {
            return std::make_unique<BackendExecutor>(descriptor, params);
        });
        descriptor.runtime = made.get();
        register_loop(std::move(made), descriptor);
    }
    return *static_cast<Loop *>(descriptor.runtime);
}

} // namespace op::rt::loop
