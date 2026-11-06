#include <op2/rt/gpu/executor.h>

#include <op2/gpu/check.h>
#include <op2/rt/gpu/layout.h>
#include <op2/rt/gpu/strategy.h>
#include <op_cuda_rt_support.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

namespace op::rt::gpu {

namespace {

#define OP_GPU_FIELD(type, name, qualifier) +1

// The larger launch record's fields.
constexpr std::size_t record_fields = std::max(0 OP_ABI_GPU_RANGE_FIELDS(OP_GPU_FIELD),
                                               0 OP_ABI_GPU_CHUNK_FIELDS(OP_GPU_FIELD));

#undef OP_GPU_FIELD

} // namespace

Executor::Executor(const abi::gpu::Loop& descriptor, op::rt::jit::ParamSet& params,
                   op::rt::Context& context)
    : m_descriptor{descriptor}, m_params{params}, m_context{context},
      m_backend{gpu::backend(context)},
      m_compiler{descriptor.name, descriptor.translated, descriptor.headers,
                 m_backend.device.arch} {
    m_params.set_dat_stride(&dat_stride);
    for (const auto& param : descriptor.params)
        add_param(param);
    m_arguments.resize(record_fields + static_cast<std::size_t>(descriptor.arguments));
}

Executor::~Executor() {
    for (auto event : m_events)
        GPU_CHECK(gpuEventDestroy(event));
}

void Executor::add_strategies(op::rt::loop::Ladder& ladder) {
    ladder.add_families(m_descriptor.families, strategy_families(), *this, "GPU");
}

void Executor::add_param(const abi::gpu::Param& param) {
    auto index = m_params.add(param);
    if (param.symbol != nullptr)
        m_symbols.bind(index, m_descriptor.calls->symbol(param.symbol), param.shared_hash);
    if (param.by_value) {
        auto arg = static_cast<std::size_t>(param.index);
        if (m_by_value.size() <= arg)
            m_by_value.resize(arg + 1, 0);
        m_by_value[arg] = 1;
    }
}

int Executor::exchange_halos(op_set set, std::span<op_arg> args) {
    return op_mpi_halo_exchanges(set, args, 2);
}

void Executor::mark_dirty(std::span<op_arg> args) { op_mpi_set_dirtybit_cuda(args); }

// --- For the strategies ---

int Executor::block_limit(const op::rt::loop::Call& call, int block_size) const {
    auto bytes = reductionBytesPerThread(call.args.data(), static_cast<int>(call.args.size()),
                                         m_descriptor.gbl_inc_atomic);
    int limit = blockLimitForReductions(bytes, block_size);
    // Without a cap the scratch is no limit, nor worth a warning.
    if (limit != INT32_MAX && bytes > 8 * 1024 && !m_warned_reductions) {
        m_warned_reductions = true;
        op_printf("Warning: kernel %s needs %zu reduction bytes per thread\n", name(), bytes);
    }
    return limit;
}

const unsigned char *Executor::on_host(const op::rt::loop::Prepared& prepared, int nargs) {
    m_on_host.assign(m_by_value.begin(), m_by_value.end());
    m_on_host.resize(static_cast<std::size_t>(nargs), 0);
    if (prepared.kernel != nullptr)
        for (int arg : static_cast<const JitKernel *>(prepared.kernel)->baked_globals())
            m_on_host[static_cast<std::size_t>(arg)] = 1;
    return m_on_host.data();
}

void Executor::prepare_globals(const op::rt::loop::Prepared& prepared,
                               const op::rt::loop::Call& call, int global_stride) {
    int nargs = static_cast<int>(call.args.size());
    prepareDeviceGblsWithPolicy(call.args.data(), nargs, global_stride,
                                m_descriptor.gbl_inc_atomic, on_host(prepared, nargs));
}

bool Executor::combine_globals(const op::rt::loop::Call& call, int global_stride) {
    return processDeviceGblsWithPolicy(call.args.data(), static_cast<int>(call.args.size()),
                                       global_stride, global_stride,
                                       m_descriptor.gbl_inc_atomic);
}

void Executor::bind(std::span<op_arg> args, unsigned& opt_flags, std::size_t fields) {
    auto arguments = std::span{m_arguments}.subspan(
        fields, static_cast<std::size_t>(m_descriptor.arguments));
    [[maybe_unused]] int written = m_descriptor.bind(args.data(), opt_flags, arguments);
    assert(written == m_descriptor.arguments);
}

#define OP_GPU_ARGUMENT(type, name, qualifier) m_arguments[fields++] = &record.name;

void **Executor::arguments(abi::gpu::Range& record, std::span<op_arg> args) {
    std::size_t fields = 0;
    OP_ABI_GPU_RANGE_FIELDS(OP_GPU_ARGUMENT)
    bind(args, record.opt_flags, fields);
    return m_arguments.data();
}

void **Executor::arguments(abi::gpu::Chunk& record, std::span<op_arg> args) {
    std::size_t fields = 0;
    OP_ABI_GPU_CHUNK_FIELDS(OP_GPU_ARGUMENT)
    bind(args, record.opt_flags, fields);
    return m_arguments.data();
}

#undef OP_GPU_ARGUMENT

gpuEvent_t Executor::event(std::size_t index) {
    while (m_events.size() <= index) {
        gpuEvent_t event;
        GPU_CHECK(gpuEventCreate(&event));
        m_events.push_back(event);
    }
    return m_events[index];
}

} // namespace op::rt::gpu

namespace op::rt::abi::gpu {

void invoke(Loop& loop, op_set set, op_arg *args, int nargs) {
    op::rt::loop::runtime<op::rt::gpu::Executor>(loop).invoke(
        set, {args, static_cast<std::size_t>(nargs)});
}

} // namespace op::rt::abi::gpu
