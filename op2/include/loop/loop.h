#pragma once

// The base of every backend's loop, the object the generated code builds per
// par_loop: its ladder, its tuner and its JIT parameters, and the steps of a
// call that are the same on every backend.  A backend derives its loop and
// runs the exchange, the globals and its profile scopes around those steps.

#include <loop/autotune.h>
#include <loop/jit.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <optional>
#include <string>
#include <string_view>

namespace op::loop {

class Loop;

// Every loop the generated code builds, which the reports read and finalise
// releases.
void register_loop(Loop *loop);
void unregister_loop(Loop *loop);

// At op_profile_output and op_exit: write the tuner's and the JIT's reports,
// OP_AUTOTUNE_REPORT and OP_JIT_REPORT, suffixed with the rank when there
// is more than one, and print their summaries of this rank's loops once, on
// rank 0.  A later call writes the files again, so applications that never
// call op_exit still get them.
void write_reports(int rank, int ranks);

// At op_exit, while the device is still up: write the reports, then release
// every loop.
void finalise(int rank, int ranks);

// The summaries write_reports prints: the tuner's when it is on, the JIT's
// under OP_JIT_REPORT when it built or observed anything; empty otherwise.
std::string summaries(int ranks);

class Loop {
protected:
    std::string m_name;
    std::string m_profile_target;
    std::string m_profile_variant;
    bool m_direct;
    Ladder m_ladder;
    std::optional<Autotuner> m_tuner;
    ParamSet m_params;

    // A call's decision: the parameters take its values, the ladder selects,
    // and the tuner, while it explores the configuration, may choose another
    // candidate.  Every call counts in the parameters' statistics, whichever
    // candidate runs, so a specialised kernel meets settled modes, and the
    // tuner waits for them.
    Autotuner::Decision decide(const Call& call, op_arg *args,
                               int nargs) {
        m_params.update(args, nargs);
        if (m_params.moving() && jit_options().enabled())
            m_params.observe_call(call);
        auto selection = m_ladder.select(call);
        return m_tuner ? m_tuner->choose(call, selection, m_params.settled(call))
                       : Autotuner::Decision{selection};
    }

    // Run a decision, its time a sample for the tuner when it is timed.
    void run(const Autotuner::Decision& decision, op_arg *args, int nargs) {
        Launch launch{args, nargs, decision.timed != nullptr};
        auto microseconds =
            decision.prepared.candidate->strategy->run(decision.prepared, launch);
        if (microseconds && m_tuner)
            m_tuner->sample(decision, *microseconds);
    }

public:
    // profile_target and profile_variant name the loop to the profiler.
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, bool direct)
        : m_name{name}, m_profile_target{profile_target},
          m_profile_variant{profile_variant}, m_direct{direct}, m_ladder{name},
          m_params{name} {
        if (autotune_options().report)
            m_tuner.emplace(m_ladder);
        register_loop(this);
    }

    virtual ~Loop() { unregister_loop(this); }
    Loop(const Loop&) = delete;

    const std::string& name() const { return m_name; }
    Ladder& ladder() { return m_ladder; }
    const Ladder& ladder() const { return m_ladder; }
    ParamSet& params() { return m_params; }
    const ParamSet& params() const { return m_params; }
    const Autotuner *tuner() const { return m_tuner ? &*m_tuner : nullptr; }

    // Its specialised kernels' builds, by where they stand; none without a
    // JIT.
    virtual BuildCounts builds() const { return {}; }

    Sections sections(const Call& call) const {
        return Sections::of(call, m_direct);
    }

    // At op_exit, while the device is still up, once the reports are
    // written: forget the tuner's configurations.  A backend also frees its
    // device state.
    virtual void release() {
        if (m_tuner)
            m_tuner->clear();
    }
};

} // namespace op::loop
