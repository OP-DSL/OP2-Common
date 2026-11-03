#include <op2/rt/gpu/loop.h>

#include <op2/gpu/check.h>
#include <op2/rt/gpu/jit.h>
#include <op2/rt/gpu/layout.h>
#include <op2/rt/gpu/strategy.h>
#include <op_cuda_rt_support.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace op::rt::gpu {

Loop::Loop(const abi::gpu::Loop& descriptor)
    : op::rt::loop::Loop{descriptor.name, descriptor.target, descriptor.variant,
                         descriptor.direct},
      m_gbl_inc_atomic{descriptor.gbl_inc_atomic}, m_bind{descriptor.bind},
      m_calls{descriptor.calls}, m_symbols{std::make_unique<SymbolBindings>()},
      m_compiler{std::make_unique<Compiler>(descriptor.name, descriptor.headers)} {
    m_params.set_dat_stride(&dat_stride);
    for (const auto& param : descriptor.params)
        add_param(param);

    for (const auto& family : descriptor.families) {
        auto make = strategy_family(family.name, descriptor.name).make;
        m_ladder.add(family.name, [&](const op::rt::loop::Point& point) {
            return make(*this, family, point);
        });
    }
}

Loop::~Loop() {
    for (auto event : m_events)
        GPU_CHECK(gpuEventDestroy(event));
}

void Loop::add_param(const abi::gpu::Param& param) {
    auto index = m_params.add(param);
    if (param.symbol != nullptr)
        m_symbols->bind(index, m_calls->symbol(param.symbol), param.shared_hash);
    if (param.by_value) {
        auto arg = static_cast<std::size_t>(param.index);
        if (m_by_value.size() <= arg)
            m_by_value.resize(arg + 1, 0);
        m_by_value[arg] = 1;
    }
}

// --- For the strategies ---

void Loop::upload() { m_symbols->upload(m_params); }

int Loop::block_limit(const op::rt::loop::Call& call, int block_size) const {
    auto bytes = reductionBytesPerThread(call.args.data(), static_cast<int>(call.args.size()),
                                         m_gbl_inc_atomic);
    int limit = blockLimitForReductions(bytes, block_size);
    // Without a cap the scratch is no limit, nor worth a warning.
    if (limit != INT32_MAX && bytes > 8 * 1024 && !m_warned_reductions) {
        m_warned_reductions = true;
        op_printf("Warning: kernel %s needs %zu reduction bytes per thread\n", m_name.c_str(),
                  bytes);
    }
    return limit;
}

#define OP_GPU_ARGUMENT(type, name, qualifier) m_arguments.push_back(&record.name);

void **Loop::arguments(abi::gpu::Range& record, std::span<op_arg> args) {
    m_arguments.clear();
    OP_ABI_GPU_RANGE_FIELDS(OP_GPU_ARGUMENT)
    m_bind(args.data(), record.opt_flags, m_arguments);
    return m_arguments.data();
}

void **Loop::arguments(abi::gpu::Chunk& record, std::span<op_arg> args) {
    m_arguments.clear();
    OP_ABI_GPU_CHUNK_FIELDS(OP_GPU_ARGUMENT)
    m_bind(args.data(), record.opt_flags, m_arguments);
    return m_arguments.data();
}

#undef OP_GPU_ARGUMENT

gpuEvent_t Loop::event(std::size_t index) {
    while (m_events.size() <= index) {
        gpuEvent_t event;
        GPU_CHECK(gpuEventCreate(&event));
        m_events.push_back(event);
    }
    return m_events[index];
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

void Loop::prepare_globals(const op::rt::loop::Prepared& prepared,
                           const op::rt::loop::Call& call, int global_stride) {
    m_global_stride = global_stride;
    m_exit_sync = false;
    int nargs = static_cast<int>(call.args.size());
    prepareDeviceGblsWithPolicy(call.args.data(), nargs, m_global_stride, m_gbl_inc_atomic,
                                on_host(prepared, nargs));
}

int Loop::exchange_halos(op_set set, std::span<op_arg> args) {
    return op_mpi_halo_exchanges(set, args, 2);
}

void Loop::mark_dirty(std::span<op_arg> args) { op_mpi_set_dirtybit_cuda(args); }

void Loop::combine_globals(const op::rt::loop::Call& call) {
    m_exit_sync |= processDeviceGblsWithPolicy(call.args.data(),
                                               static_cast<int>(call.args.size()),
                                               m_global_stride, m_global_stride, m_gbl_inc_atomic);
}

void Loop::end_call() {
    if (m_exit_sync)
        GPU_CHECK(gpuStreamSynchronize(0));
}

} // namespace op::rt::gpu

namespace op::rt::abi::gpu {

void invoke(Loop& loop, op_set set, op_arg *args, int nargs) {
    op::rt::loop::runtime<op::rt::gpu::Loop>(loop).invoke(set, args, nargs);
}

} // namespace op::rt::abi::gpu
