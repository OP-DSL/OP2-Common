#include <op2/rt/loop/loop.h>

#include <op_lib_core.h>
#include <op_profile.h>

namespace op::rt::loop {

tune::Autotuner::Decision Loop::decide(const Call& call, op_arg *args, int nargs) {
    m_params.update(args, nargs);
    if (m_params.moving() && jit::options().enabled())
        m_params.observe_call(call);
    auto selection = m_ladder.select(call);
    return m_tuner ? m_tuner->choose(call, selection, m_params.settled(call))
                   : tune::Autotuner::Decision{selection};
}

void Loop::run(const tune::Autotuner::Decision& decision, const Call& call, op_arg *args,
               int nargs) {
    RunArgs run{call, args, nargs, decision.timed != nullptr};
    auto microseconds = decision.prepared.candidate->strategy->run(decision.prepared, run);
    if (microseconds && m_tuner)
        m_tuner->sample(decision, *microseconds);
}

void Loop::wait_halos(const RunArgs& run) {
    op_profile_next("MPI Wait");
    op_mpi_wait_all(run.nargs, run.args);
    op_profile_next("Kernel");
}

void Loop::process_globals(const RunArgs& run) {
    if (!m_global_output)
        return;
    op_profile_next("Process GBLs");
    combine_globals(run.args, run.nargs);
    op_profile_next("Kernel");
}

void Loop::invoke(op_set set, op_arg *args, int nargs) {
    op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                            m_profile_variant.c_str());
    op_profile_enter("MPI Exchanges");
    int n_exec = exchange_halos(set, args, nargs);

    if (n_exec > 0) {
        op_profile_next("Init");
        Call call{set, {args, static_cast<std::size_t>(nargs)}, n_exec, m_direct};
        auto decision = decide(call, args, nargs);

        op_profile_next("Prepare GBLs");
        m_global_output = call.writes_globals();
        begin_call(call, args, nargs);
        decision.prepared.candidate->strategy->begin_call(decision.prepared, args, nargs);

        op_profile_next("Computation");
        op_profile_enter("Kernel");
        run(decision, call, args, nargs);
        op_profile_exit();
        end_call();
    } else {
        op_profile_next("MPI Wait");
        op_mpi_wait_all(nargs, args);
    }

    op_profile_next("MPI Reduce");
    reduce_mpi_globals(args, nargs, m_name.c_str());
    mark_dirty(args, nargs);
    op_profile_exit();
    op_profile_exit();
}

} // namespace op::rt::loop
