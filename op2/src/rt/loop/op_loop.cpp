#include <op2/rt/loop/loop.h>

#include <op_lib_core.h>
#include <op_profile.h>

namespace op::rt::loop {

namespace {

bool has_global_output(op_arg *args, int nargs) {
    for (int i = 0; i < nargs; ++i) {
        if (args[i].opt == 0 || args[i].argtype != OP_ARG_GBL)
            continue;
        if (args[i].acc == OP_INC || args[i].acc == OP_MIN || args[i].acc == OP_MAX ||
            args[i].acc == OP_RW || args[i].acc == OP_WRITE)
            return true;
    }
    return false;
}

} // namespace

Autotuner::Decision Loop::decide(const Call& call, op_arg *args, int nargs) {
    m_params.update(args, nargs);
    if (m_params.moving() && jit_options().enabled())
        m_params.observe_call(call);
    auto selection = m_ladder.select(call);
    return m_tuner ? m_tuner->choose(call, selection, m_params.settled(call))
                   : Autotuner::Decision{selection};
}

void Loop::run(const Autotuner::Decision& decision, op_arg *args, int nargs) {
    Launch launch{args, nargs, decision.timed != nullptr};
    auto microseconds =
        decision.prepared.candidate->strategy->run(decision.prepared, launch);
    if (microseconds && m_tuner)
        m_tuner->sample(decision, *microseconds);
}

void Loop::wait_halos(const Launch& launch) {
    op_profile_next("MPI Wait");
    op_mpi_wait_all(launch.nargs, launch.args);
    op_profile_next("Kernel");
}

void Loop::process_globals(const Launch& launch) {
    if (!m_global_output)
        return;
    op_profile_next("Process GBLs");
    combine_globals(launch.args, launch.nargs);
    op_profile_next("Kernel");
}

void Loop::invoke(op_set set, op_arg *args, int nargs) {
    op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                            m_profile_variant.c_str());
    op_profile_enter("MPI Exchanges");
    int n_exec = op_mpi_halo_exchanges(set, nargs, args, m_device ? 2 : 1);

    if (n_exec > 0) {
        op_profile_next("Init");
        Call call{set, {args, static_cast<std::size_t>(nargs)}, n_exec};
        auto decision = decide(call, args, nargs);

        op_profile_next("Prepare GBLs");
        m_global_output = has_global_output(args, nargs);
        begin_call(call, decision, args, nargs);

        op_profile_next("Computation");
        op_profile_enter("Kernel");
        run(decision, args, nargs);
        op_profile_exit();
        end_call();
    } else {
        op_profile_next("MPI Wait");
        op_mpi_wait_all(nargs, args);
    }

    op_profile_next("MPI Reduce");
    reduce_mpi_globals(args, nargs, m_name.c_str());
    if (m_device)
        op_mpi_set_dirtybit_cuda(nargs, args);
    else
        op_mpi_set_dirtybit(nargs, args);
    op_profile_exit();
    op_profile_exit();
}

} // namespace op::rt::loop
