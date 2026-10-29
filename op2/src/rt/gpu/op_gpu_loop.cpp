#include <op2/rt/gpu/loop.h>

#include <op2/gpu/check.h>
#include <op2/rt/gpu/jit.h>
#include <op2/rt/gpu/strategy.h>
#include <op_cuda_rt_support.h>

#include <utility>

namespace op::rt::gpu {

Loop::Loop(std::string_view name, std::string_view profile_target,
           std::string_view profile_variant, bool direct, bool gbl_inc_atomic,
           JitHeaders headers)
    : op::rt::loop::Loop{name, profile_target, profile_variant, direct, true},
      m_gbl_inc_atomic{gbl_inc_atomic}, m_symbols{std::make_unique<SymbolBindings>()},
      m_compiler{std::make_unique<Compiler>(name, std::move(headers))} {}

Loop::~Loop() { release(); }

// --- Registration ---

void Loop::bind(std::size_t param, void *symbol_d, std::uint64_t *shared_hash) {
    if (symbol_d != nullptr)
        m_symbols->bind(param, symbol_d, shared_hash);
}

void Loop::mark_by_value(int arg_index) {
    if (m_by_value.size() <= static_cast<std::size_t>(arg_index))
        m_by_value.resize(static_cast<std::size_t>(arg_index) + 1, 0);
    m_by_value[static_cast<std::size_t>(arg_index)] = 1;
}

void Loop::add(std::string_view family, KernelSource source, Binder bind,
               op::rt::plan::ArgGroups groups) {
    m_ladder.add(family, [&](const op::rt::loop::Point& point) {
        return make_strategy(*this, family, point, source, bind, groups);
    });
}

void Loop::add_dat_stride_param(std::string_view name, int *data, int arg_index,
                                void *symbol_d) {
    bind(m_params.add(name, data, op::rt::loop::ParamSource::dat_stride, arg_index, true),
         symbol_d, nullptr);
}

void Loop::add_dat_dim_param(std::string_view name, int *data, int arg_index) {
    m_params.add(name, data, op::rt::loop::ParamSource::dat_dim, arg_index);
}

void Loop::add_launch_param(std::string_view name, LaunchValue value, void *symbol_d) {
    bind(m_params.add(name, static_cast<int *>(nullptr), op::rt::loop::ParamSource::strategy,
                      static_cast<int>(value)),
         symbol_d, nullptr);
}

// --- For the strategies ---

void Loop::upload(const LaunchValues& launch) {
    m_params.apply(launch.slots());
    m_symbols->upload(m_params);
}

int Loop::block_limit(op_arg *args, int nargs, int block_size) const {
    return getBlockLimitWithPolicy(args, nargs, block_size, m_name.c_str(), m_gbl_inc_atomic);
}

gpuEvent_t Loop::event(std::size_t index) {
    while (m_events.size() <= index) {
        gpuEvent_t event;
        GPU_CHECK(gpuEventCreate(&event));
        m_events.push_back(event);
    }
    return m_events[index];
}

void Loop::release() {
    op::rt::loop::Loop::release();
    for (auto event : m_events)
        GPU_CHECK(gpuEventDestroy(event));
    m_events.clear();
}

// --- Per call ---

const unsigned char *Loop::on_host(const op::rt::loop::Prepared& prepared, int nargs) {
    m_on_host.assign(m_by_value.begin(), m_by_value.end());
    m_on_host.resize(static_cast<std::size_t>(nargs), 0);
    if (prepared.kernel != nullptr)
        for (int arg : static_cast<const JitKernel *>(prepared.kernel)->baked_globals())
            m_on_host[static_cast<std::size_t>(arg)] = 1;
    return m_on_host.data();
}

void Loop::begin_call(const op::rt::loop::Call&,
                      const op::rt::loop::Autotuner::Decision& decision, op_arg *args,
                      int nargs) {
    const auto& state =
        static_cast<StrategyBase *>(decision.prepared.candidate->strategy)->call_state();
    m_global_stride = state.global_stride;
    m_exit_sync = false;
    prepareDeviceGblsWithPolicy(args, nargs, m_global_stride, m_gbl_inc_atomic,
                                on_host(decision.prepared, nargs));
}

void Loop::combine_globals(op_arg *args, int nargs) {
    m_exit_sync |= processDeviceGblsWithPolicy(args, nargs, m_global_stride, m_global_stride,
                                               m_gbl_inc_atomic);
}

void Loop::end_call() {
    if (m_exit_sync)
        GPU_CHECK(gpuStreamSynchronize(0));
}

} // namespace op::rt::gpu
