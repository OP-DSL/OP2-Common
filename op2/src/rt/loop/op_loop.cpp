#include <op2/rt/loop/loop.h>

#include <op_lib_core.h>
#include <op_profile.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace op::rt::loop {

tune::Autotuner::Decision Loop::decide(const Call& call) {
    m_params.update(call.args);
    if (m_jit.enabled())
        m_params.observe(call);
    return m_tuner ? m_tuner->choose(call, m_params.settled(call))
                   : tune::Autotuner::Decision{m_ladder.select(call)};
}

void Loop::run(const tune::Autotuner::Decision& decision, const Call& call) {
    auto microseconds = decision.prepared.candidate->strategy->run(decision.prepared, call,
                                                                   decision.timed != nullptr);
    if (microseconds && m_tuner)
        m_tuner->sample(decision, *microseconds);
}

void wait_halos(const Call& call) {
    op_profile_next("MPI Wait");
    op_mpi_wait_all(call.args);
    op_profile_next("Kernel");
}

void Loop::check_dats(std::span<const op_arg> args) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT || arg.dat == nullptr)
            continue;
        auto type = m_types[i];
        bool word = type == abi::ElementType::f32 || type == abi::ElementType::i32;
        int size = type == abi::ElementType::f64 ? 8 : word ? 4 : 0;
        if (arg.dim != arg.dat->dim || (size != 0 && size * arg.dim != arg.dat->size)) {
            std::fprintf(stderr,
                         "error: argument %zu of %s passes dat %s, of dim %d and %d-byte "
                         "elements, as dim %d of %s\n",
                         i, m_name.c_str(), arg.dat->name, arg.dat->dim,
                         arg.dat->size / arg.dat->dim, arg.dim, arg.type);
            std::exit(1);
        }
    }
}

void Loop::invoke(op_set set, std::span<op_arg> args) {
    if (m_types.size() != args.size())
        for (const auto& arg : args) {
            std::string type = arg.type != nullptr ? arg.type : "";
            if (type.ends_with(":soa"))
                type.resize(type.size() - 4);
            m_types.push_back(element_type(type.c_str()));
        }
    check_dats(args);
    op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                            m_profile_variant.c_str());
    op_profile_enter("MPI Exchanges");
    int n_exec = m_executor->exchange_halos(set, args);

    if (n_exec > 0) {
        op_profile_next("Init");
        const Call call{set, args, n_exec, m_direct};
        auto decision = decide(call);

        op_profile_next("Computation");
        op_profile_enter("Kernel");
        run(decision, call);
        op_profile_exit();
    } else {
        op_profile_next("MPI Wait");
        op_mpi_wait_all(args);
    }

    op_profile_next("MPI Reduce");
    reduce_mpi_globals(args, m_types, m_name.c_str());
    m_executor->mark_dirty(args);
    op_profile_exit();
    op_profile_exit();
}

} // namespace op::rt::loop
