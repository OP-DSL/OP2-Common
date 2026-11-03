#include <op2/rt/loop/loop.h>

#include <op_lib_core.h>
#include <op_profile.h>

namespace op::rt::loop {

tune::Autotuner::Decision Loop::decide(const Call& call) {
    m_params.update(call.args);
    if (m_params.moving() && jit::options().enabled())
        m_params.observe_call(call);
    return m_tuner ? m_tuner->choose(call, m_params.settled(call))
                   : tune::Autotuner::Decision{m_ladder.select(call)};
}

void Loop::run(const tune::Autotuner::Decision& decision, const Call& call) {
    RunArgs run{call, decision.timed != nullptr};
    auto microseconds = decision.prepared.candidate->strategy->run(decision.prepared, run);
    if (microseconds && m_tuner)
        m_tuner->sample(decision, *microseconds);
}

void Loop::wait_halos(const RunArgs& run) {
    op_profile_next("MPI Wait");
    op_mpi_wait_all(run.call.args);
    op_profile_next("Kernel");
}

void Loop::process_globals(const RunArgs& run) {
    if (!run.call.writes_globals())
        return;
    op_profile_next("Process GBLs");
    combine_globals(run.call);
    op_profile_next("Kernel");
}

void Loop::invoke(op_set set, op_arg *args, int nargs) {
    op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                            m_profile_variant.c_str());
    op_profile_enter("MPI Exchanges");
    std::span<op_arg> arguments{args, static_cast<std::size_t>(nargs)};
    int n_exec = exchange_halos(set, arguments);

    if (n_exec > 0) {
        op_profile_next("Init");
        Call call{set, arguments, n_exec, m_direct};
        auto decision = decide(call);

        op_profile_next("Prepare GBLs");
        begin_call(call);
        decision.prepared.candidate->strategy->begin_call(decision.prepared, call);

        op_profile_next("Computation");
        op_profile_enter("Kernel");
        run(decision, call);
        op_profile_exit();
        end_call();
    } else {
        op_profile_next("MPI Wait");
        op_mpi_wait_all(arguments);
    }

    op_profile_next("MPI Reduce");
    reduce_mpi_globals(arguments, m_name.c_str());
    mark_dirty(arguments);
    op_profile_exit();
    op_profile_exit();
}

} // namespace op::rt::loop
