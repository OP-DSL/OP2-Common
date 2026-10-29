#pragma once

// The base of every backend's loop, the object the generated code builds per
// par_loop: its ladder, its tuner and its JIT parameters, and a call's steps,
// phases and profile scopes, from src/rt/loop/op_loop.cpp.  A backend
// derives its loop, prepares, combines and finishes the call's globals, and
// times its strategies' phases.

#include <op2/rt/jit/params.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/ladder.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/tune/autotune.h>
#include <op_lib_core.h>

#include <optional>
#include <string>
#include <string_view>

namespace op::rt::loop {

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
private:
    // A call's decision: the parameters take its values, the ladder selects,
    // and the tuner, while it explores the configuration, may choose another
    // candidate.  Every call counts in the parameters' statistics, whichever
    // candidate runs, so a specialised kernel meets settled modes, and the
    // tuner waits for them.
    tune::Autotuner::Decision decide(const Call& call, op_arg *args, int nargs);

    // Run a decision, its time a sample for the tuner when it is timed.
    void run(const tune::Autotuner::Decision& decision, const Call& call, op_arg *args,
             int nargs);

    // Whether the call under way writes a global.
    bool m_global_output = false;

    // Wait for the call's halo exchanges.
    void wait_halos(const RunArgs& run);

    // Combine the call's global reductions, if it writes any.
    void process_globals(const RunArgs& run);

protected:
    std::string m_name;
    std::string m_profile_target;
    std::string m_profile_variant;
    bool m_direct;
    // Whether the data is on the device, for the exchange and the dirty bits.
    bool m_device;
    Ladder m_ladder;
    std::optional<tune::Autotuner> m_tuner;
    jit::ParamSet m_params;

    // Prepare the call's globals for the decision's run.
    virtual void begin_call(const Call&, const tune::Autotuner::Decision&, op_arg *, int) {}

    // Combine the run's partial reductions into the arguments, once the
    // elements they cover have run.
    virtual void combine_globals(op_arg *, int) {}

    // Finish the run's globals before the host reduces them.
    virtual void end_call() {}

public:
    // profile_target and profile_variant name the loop to the profiler.
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, bool direct, bool device)
        : m_name{name}, m_profile_target{profile_target},
          m_profile_variant{profile_variant}, m_direct{direct}, m_device{device},
          m_ladder{name}, m_params{name} {
        if (tune::autotune_options().report)
            m_tuner.emplace(m_ladder);
        register_loop(this);
    }

    virtual ~Loop() { unregister_loop(this); }
    Loop(const Loop&) = delete;

    const std::string& name() const { return m_name; }
    bool direct() const { return m_direct; }
    Ladder& ladder() { return m_ladder; }
    const Ladder& ladder() const { return m_ladder; }
    jit::ParamSet& params() { return m_params; }
    const jit::ParamSet& params() const { return m_params; }
    const tune::Autotuner *tuner() const { return m_tuner ? &*m_tuner : nullptr; }

    // Its specialised kernels' builds, by where they stand.
    virtual BuildCounts builds() const {
        BuildCounts counts;
        for (const auto& strategy : m_ladder.strategies())
            counts += strategy->builds();
        return counts;
    }

    // Run one call: start the halo exchange, which decides what the call runs
    // and which the planning overlaps; decide; prepare the globals; run; and
    // finish, reduce and mark the globals and dats.
    void invoke(op_set set, op_arg *args, int nargs);

    // Run a strategy's phases, phase(index) each: the halo wait before phase
    // wait_before, or after the last when it is phases, and the globals'
    // combine after phase globals_after.
    template<typename Phase>
    void run_phases(const RunArgs& run, int phases, int wait_before, int globals_after,
                    Phase&& phase) {
        for (int index = 0; index < phases; ++index) {
            if (index == wait_before)
                wait_halos(run);
            phase(index);
            if (index == globals_after)
                process_globals(run);
        }
        if (phases == wait_before)
            wait_halos(run);
    }

    // At op_exit, while the device is still up, once the reports are
    // written: forget the tuner's configurations and release the strategies.
    // A backend also frees its own device state.
    virtual void release() {
        if (m_tuner)
            m_tuner->clear();
        for (const auto& strategy : m_ladder.strategies())
            strategy->release();
    }
};

} // namespace op::rt::loop
