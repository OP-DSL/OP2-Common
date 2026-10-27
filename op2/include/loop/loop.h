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

class Loop {
private:
    static void release_callback(void *owner) { static_cast<Loop *>(owner)->release(); }

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
    // strategy.  Every call counts in the parameters' statistics, whichever
    // strategy runs, so one that specialises later meets settled modes.
    Autotuner::Decision decide(const Call& call, op_arg *args,
                               int nargs) {
        m_params.update(args, nargs);
        if (m_params.moving() && jit_options().enabled())
            m_params.observe_call(call);
        auto selection = m_ladder.select(call);
        return m_tuner ? m_tuner->choose(call, selection)
                       : Autotuner::Decision{selection};
    }

    // Run a decision, its time a sample for the tuner when it is timed.
    void run(const Autotuner::Decision& decision, op_arg *args, int nargs) {
        Launch launch{args, nargs, decision.timed != nullptr};
        auto microseconds = decision.prepared.strategy->run(decision.prepared, launch);
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
        register_release(this, &Loop::release_callback);
    }

    virtual ~Loop() { unregister_release(this); }
    Loop(const Loop&) = delete;

    const std::string& name() const { return m_name; }
    Ladder& ladder() { return m_ladder; }
    ParamSet& params() { return m_params; }

    Sections sections(const Call& call) const {
        return Sections::of(call, m_direct);
    }

    // At op_exit, while the device is still up: report the configurations
    // still being explored.  A backend also frees its device state.
    virtual void release() {
        if (m_tuner)
            m_tuner->flush();
    }
};

} // namespace op::loop
