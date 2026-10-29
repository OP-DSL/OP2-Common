#include <op2/rt/gpu/loop.h>

#include <op2/gpu/check.h>
#include <op2/rt/gpu/jit.h>
#include <op2/rt/gpu/strategy.h>
#include <op_cuda_rt_support.h>

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace op::rt::gpu {

Loop::Loop(const abi::gpu::Loop& descriptor)
    : op::rt::loop::Loop{descriptor.name, descriptor.target, descriptor.variant,
                         descriptor.direct, true},
      m_gbl_inc_atomic{descriptor.gbl_inc_atomic}, m_bind{descriptor.bind},
      m_calls{descriptor.calls}, m_symbols{std::make_unique<SymbolBindings>()},
      m_compiler{std::make_unique<Compiler>(descriptor.name, descriptor.headers)} {
    for (const auto& param : descriptor.params)
        add_param(param);

    // The strategies' values, each a constant of a specialised kernel: the
    // global stride when a kernel keeps a copy per thread of some global, and
    // the chunked strategies' switches when the loop has one.
    auto strategy_param = [&](const char *name, LaunchValue value) {
        m_params.add(name, static_cast<int *>(nullptr), abi::ParamSource::strategy,
                     static_cast<int>(value));
    };
    strategy_param("op2_block_size_d", LaunchValue::block_size);
    if (descriptor.per_thread_globals)
        strategy_param("op2_stride_gbl_d", LaunchValue::global_stride);
    bool chunked = false;
    for (const auto& family : descriptor.families)
        chunked |= std::string_view{family.name}.starts_with("chunk_");
    if (chunked) {
        strategy_param("op2_unstaged_singles_d", LaunchValue::unstaged_singles);
        strategy_param("op2_wavefront_d", LaunchValue::wavefront);
    }

    for (const auto& family : descriptor.families)
        m_ladder.add(family.name, [&](const op::rt::loop::Point& point) {
            return make_strategy(*this, family, point);
        });
}

Loop::~Loop() { release(); }

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

int Loop::block_limit(op_arg *args, int nargs, int block_size) const {
    return getBlockLimitWithPolicy(args, nargs, block_size, m_name.c_str(), m_gbl_inc_atomic);
}

#define OP_GPU_ARGUMENT(type, name, qualifier) m_arguments.push_back(&record.name);

void **Loop::arguments(abi::gpu::Range& record, op_arg *args) {
    m_arguments.clear();
    OP_ABI_GPU_RANGE_FIELDS(OP_GPU_ARGUMENT)
    m_bind(args, record.opt_flags, m_arguments);
    return m_arguments.data();
}

void **Loop::arguments(abi::gpu::Chunk& record, op_arg *args) {
    m_arguments.clear();
    OP_ABI_GPU_CHUNK_FIELDS(OP_GPU_ARGUMENT)
    m_bind(args, record.opt_flags, m_arguments);
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
                      const op::rt::tune::Autotuner::Decision& decision, op_arg *args,
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

namespace op::rt::abi::gpu {

namespace {

// The loops made, which live as long as the program.
std::vector<std::unique_ptr<op::rt::gpu::Loop>>& loops() {
    static std::vector<std::unique_ptr<op::rt::gpu::Loop>> made;
    return made;
}

} // namespace

void invoke(Loop& loop, op_set set, op_arg *args, int nargs) {
    auto *runtime = static_cast<op::rt::gpu::Loop *>(loop.runtime);
    if (runtime == nullptr) {
        runtime = loops().emplace_back(std::make_unique<op::rt::gpu::Loop>(loop)).get();
        loop.runtime = runtime;
    }
    runtime->invoke(set, args, nargs);
}

} // namespace op::rt::abi::gpu
