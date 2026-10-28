#pragma once

// The CUDA and HIP backend's loop: the object the generated code builds per
// par_loop, from plain data, adding to the base loop the parameters' device
// symbols, the compiler of its specialised kernels and the launches, and
// running one call through them.

#include <gpu/check.h>
#include <gpu/jit.h>
#include <gpu/strategy.h>
#include <loop/loop.h>
#include <loop/strategy.h>
#include <op_cuda_rt_support.h>
#include <op_lib_cpp.h>
#include <op_profile.h>
#include <plan/options.h>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::gpu {

class Loop : public op::loop::Loop {
private:
    bool m_gbl_inc_atomic;
    SymbolBindings m_symbols;
    Compiler m_compiler;
    // Per argument, whether every kernel has it by value; and the call's
    // flags, which add the READ globals the chosen kernel bakes in.
    std::vector<unsigned char> m_by_value;
    std::vector<unsigned char> m_on_host;
    // Event pairs bracketing each phase's launches of a timed call.
    std::vector<gpuEvent_t> m_events;
    // A launch's kernel arguments, filled by the binders.
    std::vector<void *> m_arguments;
    // The call under way: whether any global is written, and whether its
    // processing needs a synchronisation before the host reads.
    bool m_global_output = false;
    bool m_exit_sync = false;

    static bool has_global_output(op_arg *args, int nargs) {
        for (int i = 0; i < nargs; ++i) {
            if (args[i].opt == 0 || args[i].argtype != OP_ARG_GBL)
                continue;
            if (args[i].acc == OP_INC || args[i].acc == OP_MIN ||
                args[i].acc == OP_MAX || args[i].acc == OP_RW ||
                args[i].acc == OP_WRITE)
                return true;
        }
        return false;
    }

    // Bind a parameter to the device symbol an offline kernel reads it from,
    // if there is one.
    template<typename T>
    void bind(std::size_t param, const T *symbol, std::uint64_t *shared_hash) {
        if (symbol == nullptr)
            return;
        void *data_d = nullptr;
        GPU_CHECK(gpuGetSymbolAddress(&data_d, (const void *)symbol));
        m_symbols.bind(param, data_d, shared_hash);
    }

    void mark_by_value(int arg_index) {
        if (m_by_value.size() <= static_cast<std::size_t>(arg_index))
            m_by_value.resize(static_cast<std::size_t>(arg_index) + 1, 0);
        m_by_value[static_cast<std::size_t>(arg_index)] = 1;
    }

    // Per argument, whether the kernel that runs has it from the host side
    // and so needs no device copy of it.
    const unsigned char *on_host(const op::loop::Prepared& prepared, int nargs) {
        m_on_host.assign(m_by_value.begin(), m_by_value.end());
        m_on_host.resize(static_cast<std::size_t>(nargs), 0);
        if (prepared.kernel != nullptr)
            for (int arg : static_cast<const JitKernel *>(prepared.kernel)->baked_globals())
                m_on_host[static_cast<std::size_t>(arg)] = 1;
        return m_on_host.data();
    }

public:
    // profile_target and profile_variant name the loop to the profiler.
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, LoopDescription description,
         JitHeaders headers)
        : op::loop::Loop{name, profile_target, profile_variant, description.direct},
          m_gbl_inc_atomic{description.gbl_inc_atomic},
          m_compiler{name, std::move(headers)} {}

    ~Loop() override { release(); }

    // --- Registration, by the generated code ---

    // Add a family: a strategy S per point of its axes, built from the
    // loop, the point and args.
    template<typename S, typename... Args>
    void add(std::string_view family, Args&&... args) {
        m_ladder.add(family, [&](const op::loop::Point& point) {
            return std::make_unique<S>(*this, point, args...);
        });
    }

    // The parameters the kernels are specialised on.  symbol is the copy an
    // offline kernel reads, if any, and hash_device_ptr the hash of its value
    // there, shared by every loop that reads it.  A built kernel can have
    // each at run time but a dat's dimension and a strategy's value, which
    // its source needs as constants.

    template<typename T>
    void add_param(std::string_view name, T *data, const T *symbol = nullptr,
                   uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add(name, data, op::loop::ParamSource::external, -1, true),
             symbol, hash_device_ptr);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, std::size_t len,
                   const T *symbol = nullptr, uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add_array(name, data, len, op::loop::ParamSource::external, -1,
                                true),
             symbol, hash_device_ptr);
    }

    // A scalar READ global, which every kernel has by value.
    template<typename T>
    void add_scalar_arg_param(std::string_view name, T *data, int arg_index) {
        m_params.add(name, data, op::loop::ParamSource::scalar_arg, arg_index, true);
        mark_by_value(arg_index);
    }

    // Any other READ global: by value, or else read by an offline kernel
    // from its device copy.
    template<typename T>
    void add_global_param(std::string_view name, int arg_index,
                          bool by_value = false) {
        m_params.add_array(name, static_cast<T *>(nullptr), 0,
                           op::loop::ParamSource::array_arg, arg_index, true);
        if (by_value)
            mark_by_value(arg_index);
    }

    void add_dat_stride_param(std::string_view name, int *data, int arg_index,
                              const int *symbol) {
        bind(m_params.add(name, data, op::loop::ParamSource::dat_stride, arg_index,
                          true),
             symbol, nullptr);
    }

    void add_dat_dim_param(std::string_view name, int *data, int arg_index) {
        m_params.add(name, data, op::loop::ParamSource::dat_dim, arg_index);
    }

    // A value each strategy decides, given to every acquire and upload.
    void add_launch_param(std::string_view name, LaunchValue value,
                          const int *symbol = nullptr) {
        bind(m_params.add(name, static_cast<int *>(nullptr),
                          op::loop::ParamSource::strategy, static_cast<int>(value)),
             symbol, nullptr);
    }

    // --- For the strategies ---

    const Compiler& compiler() const { return m_compiler; }

    // Upload the changed values an offline kernel reads, under a strategy's
    // launch values.
    void upload(const LaunchValues& launch) {
        m_params.apply(launch.slots());
        m_symbols.upload(m_params);
    }

    // The most blocks a launch may use: as many as the reduction scratch
    // allows.
    int block_limit(op_arg *args, int nargs, int block_size) const {
        return getBlockLimitWithPolicy(args, nargs, block_size, m_name.c_str(),
                                       m_gbl_inc_atomic);
    }

    std::vector<void *>& arguments() { return m_arguments; }

    gpuEvent_t event(std::size_t index) {
        while (m_events.size() <= index) {
            gpuEvent_t event;
            GPU_CHECK(gpuEventCreate(&event));
            m_events.push_back(event);
        }
        return m_events[index];
    }

    // Process the global reductions after the elements they cover have run.
    void process_globals(op_arg *args, int nargs, int global_stride) {
        if (!m_global_output)
            return;
        op_profile_next("Process GBLs");
        m_exit_sync |= processDeviceGblsWithPolicy(args, nargs, global_stride,
                                                   global_stride, m_gbl_inc_atomic);
        op_profile_next("Kernel");
    }

    // --- Per call ---

    // Run the loop.
    void invoke(op_set set, op_arg *args, int nargs) {
        op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                                m_profile_variant.c_str());
        op_profile_enter("Init");

        // The exchange decides what the call runs, so it comes first, and
        // the planning overlaps it.
        op_profile_enter("MPI Exchanges");
        int n_exec = op_mpi_halo_exchanges(set, nargs, args, 2);
        if (n_exec == 0) {
            op_profile_exit();
            op_profile_exit();
            op_mpi_wait_all(nargs, args);
            op::loop::reduce_mpi_globals(args, nargs, m_name.c_str());
            op_mpi_set_dirtybit_cuda(nargs, args);
            op_profile_exit();
            return;
        }

        op_profile_next("Get Kernel");
        op::loop::Call call{
            set, {args, static_cast<std::size_t>(nargs)}, n_exec};
        auto decision = decide(call, args, nargs);
        const auto& state =
            static_cast<StrategyBase *>(decision.prepared.candidate->strategy)->call_state();
        op_profile_exit();

        op_profile_enter("Prepare GBLs");
        m_global_output = has_global_output(args, nargs);
        m_exit_sync = false;
        prepareDeviceGblsWithPolicy(args, nargs, state.global_stride, m_gbl_inc_atomic,
                                    on_host(decision.prepared, nargs));
        op_profile_exit();

        op_profile_next("Computation");
        op_profile_enter("Kernel");
        run(decision, args, nargs);
        op_profile_exit();
        op_profile_exit();

        op_profile_enter("Finalise");
        if (m_exit_sync)
            GPU_CHECK(gpuStreamSynchronize(0));
        op::loop::reduce_mpi_globals(args, nargs, m_name.c_str());
        op_mpi_set_dirtybit_cuda(nargs, args);
        op_profile_exit();
        op_profile_exit();
    }

    op::loop::BuildCounts builds() const override {
        op::loop::BuildCounts counts;
        for (const auto& strategy : m_ladder.strategies())
            counts += static_cast<const StrategyBase&>(*strategy).builds();
        return counts;
    }

    // At op_exit, while the device is still up: forget the tuner's
    // configurations, join the compiles, and free every strategy's device
    // state.
    void release() override {
        op::loop::Loop::release();
        for (const auto& strategy : m_ladder.strategies())
            static_cast<StrategyBase&>(*strategy).release();
        for (auto event : m_events)
            GPU_CHECK(gpuEventDestroy(event));
        m_events.clear();
    }
};

// --- The strategies' definitions that need the loop ---------------------------------

inline StrategyBase::StrategyBase(Loop& loop, const op::loop::Point& point,
                                  KernelSource source, Binder bind)
    : m_loop{loop}, m_name{point.name}, m_source{source}, m_bind{bind},
      m_block_size{point.value("block_size")},
      m_switches{point.value("launch_bounds") != 0, point.value("min_blocks"),
                 point.value("smem_spilling", 0) != 0},
      m_offline{source.offline}, m_cache{source.name, options().build_threads} {}

inline void StrategyBase::begin(const op::loop::Call& call) {
    m_call_state = {};
    m_call_state.sections = m_loop.sections(call);
    m_call_state.block_limit = m_loop.block_limit(
        const_cast<op_arg *>(call.args.data()),
        static_cast<int>(call.args.size()), m_block_size);
}

inline op::loop::Prepared StrategyBase::acquire(
    const op::loop::Call& call, op::loop::Urgency urgency,
    op::loop::KernelKind kind, const void *entry) {
    m_call_state.global_stride = m_block_size * m_call_state.max_blocks;
    op::loop::Prepared prepared;
    prepared.entry = entry;

    if (kind == op::loop::KernelKind::offline) {
        prepared.status = op::loop::Status::ready;
        return prepared;
    }

    auto& params = m_loop.params();
    auto key = params.key(call, launch_values().slots());
    auto acquired = m_cache.acquire(key, urgency, [&] {
        return m_loop.compiler().job(params, m_source, m_switches);
    });
    // Without launch bounds its registers may leave too few for its block.
    if (acquired.status == op::loop::Status::ready &&
        acquired.kernel->figures().max_threads < m_block_size) {
        prepared.reason = "thread_limit";
        return prepared;
    }
    prepared.status = acquired.status;
    prepared.kernel = acquired.kernel;
    prepared.reason = acquired.reason;
    return prepared;
}

inline void StrategyBase::launch(const op::loop::Prepared& prepared,
                                 LaunchContext& context, op_arg *args,
                                 Geometry geometry) {
    auto& arguments = m_loop.arguments();
    m_bind(context, args, arguments);
    if (prepared.kernel != nullptr) {
        op_profile_next("JIT Kernel");
        static_cast<JitKernel *>(prepared.kernel)
            ->launch(geometry.blocks, m_block_size, arguments.data(),
                     geometry.shared_bytes, m_loop.params());
        return;
    }

    op_profile_next("Offline Kernel");
    m_offline.launch(geometry.blocks, m_block_size, arguments.data(),
                     geometry.shared_bytes);
}

template<typename Launches, typename Fill>
std::optional<float> StrategyBase::run_phases(
    const op::loop::Prepared& prepared, const op::loop::Launch& launch,
    int phases, int wait_before, int globals_after, Launches&& launches,
    Fill&& fill) {
    assert(prepared.candidate->strategy == this);
    // An offline kernel reads the call's and this strategy's values from
    // device symbols.
    if (prepared.kernel == nullptr)
        m_loop.upload(launch_values());
    int timed_pairs = 0;
    for (int phase = 0; phase < phases; ++phase) {
        if (phase == wait_before) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(launch.nargs, launch.args);
            op_profile_next("Kernel");
        }

        int count = launches(phase);
        bool timing = launch.timed && count > 0;
        if (timing)
            GPU_CHECK(gpuEventRecord(m_loop.event(2 * timed_pairs), 0));

        for (int index = 0; index < count; ++index) {
            LaunchContext context;
            context.global_stride = m_call_state.global_stride;
            context.set_stride = m_call_state.sections.set_stride();
            Geometry geometry = fill(phase, index, context);
            if (geometry.blocks > 0)
                this->launch(prepared, context, launch.args, geometry);
        }

        if (timing)
            GPU_CHECK(gpuEventRecord(m_loop.event(2 * timed_pairs++ + 1), 0));
        if (phase == globals_after)
            m_loop.process_globals(launch.args, launch.nargs, m_call_state.global_stride);
    }

    if (phases == wait_before) {
        op_profile_next("MPI Wait");
        op_mpi_wait_all(launch.nargs, launch.args);
        op_profile_next("Kernel");
    }

    if (!launch.timed)
        return std::nullopt;
    if (timed_pairs == 0)
        return 0.0f;
    GPU_CHECK(gpuEventSynchronize(m_loop.event(2 * timed_pairs - 1)));
    float total = 0.0f;
    for (int pair = 0; pair < timed_pairs; ++pair) {
        float ms = 0.0f;
        GPU_CHECK(gpuEventElapsedTime(&ms, m_loop.event(2 * pair),
                                      m_loop.event(2 * pair + 1)));
        total += ms;
    }
    return total * 1000.0f;
}

// --- Plain and atomics ---

inline std::optional<float> Simple::run(const op::loop::Prepared& prepared,
                                        const op::loop::Launch& launch) {
    const auto& sections = m_call_state.sections;
    return run_phases(
        prepared, launch, sections.size(), sections.wait_index(),
        sections.globals_index(),
        [&](int phase) { return sections[phase].size() > 0 ? 1 : 0; },
        [&](int phase, int, LaunchContext& context) {
            context.start = sections[phase].begin;
            context.end = sections[phase].end;
            return Geometry{grid(sections[phase].size()), 0};
        });
}

// --- Colouring ---

inline op::loop::Prepared Colouring::prepare(
    const op::loop::Call& call, op::loop::Urgency urgency,
    op::loop::KernelKind kind) {
    begin(call);
    const auto& entry = m_plans.get(call, [&] {
        op::plan::PlanKey key{"colouring", call};
        return colouring_plans().get(key, [&] {
            auto plan = std::make_shared<ColouringPlan>();
            plan->colouring = op::plan::colour_set(call.args, m_call_state.sections);
            const auto& order = plan->colouring.order;
            if (!order.empty()) {
                GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&plan->order_d),
                                    order.size() * sizeof(int)));
                GPU_CHECK(gpuMemcpy(plan->order_d, order.data(),
                                    order.size() * sizeof(int), gpuMemcpyHostToDevice));
            }
            if (OP_diags > 3)
                op_printf("colouring: %s %d colours, %d over owned elements\n",
                            m_loop.name().c_str(), plan->colouring.colours,
                            plan->colouring.owned_colours);
            return plan;
        });
    });

    const auto& colouring = entry.colouring;
    for (int colour = 0; colour < colouring.colours; ++colour)
        m_call_state.max_blocks = std::max(
            m_call_state.max_blocks,
            grid(colouring.offsets[static_cast<std::size_t>(colour) + 1] -
                 colouring.offsets[static_cast<std::size_t>(colour)]));
    return acquire(call, urgency, kind, &entry);
}

inline std::optional<float> Colouring::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& entry = *static_cast<const ColouringPlan *>(prepared.entry);
    const auto& colouring = entry.colouring;
    // The core's colours come first, and run while the halo arrives.
    return run_phases(
        prepared, launch, colouring.colours, colouring.core_colours,
        colouring.owned_colours - 1,
        [](int) { return 1; },
        [&](int colour, int, LaunchContext& context) {
            context.start = colouring.offsets[static_cast<std::size_t>(colour)];
            context.end = colouring.offsets[static_cast<std::size_t>(colour) + 1];
            context.colour_order = entry.order_d;
            return Geometry{grid(context.end - context.start), 0};
        });
}

inline std::string Colouring::statistics(const op::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const ColouringPlan *>(prepared.entry);
    return StrategyBase::statistics(prepared) + ", " +
           std::to_string(entry.colouring.colours) + " colours";
}

inline void Colouring::release() {
    StrategyBase::release();
    m_plans.clear();
}

// --- The chunked strategies ---

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, groups, chunk size, block and variant.
inline const ChunkEntry& ChunkBase::entry_for(const op::loop::Call& call) {
    return m_plans.get(call, [&] {
        ChunkLimits limits{m_block_size, device().shared_bytes};
        const auto& chunk_options = op::plan::chunk_options();
        op::plan::PlanKey key{m_coloured ? "chunk_colouring" : "chunk_atomics", call};
        key.add(m_groups).add(m_chunk_size).add(limits.block_size).add(limits.shared_bytes);
        if (m_coloured)
            key.add(m_wavefront).add(options().window);
        else
            key.add(m_unstaged_singles).add(chunk_options.min_compression);
        return chunk_plans().get(key, [&] {
            auto result =
                m_coloured
                    ? plan_chunk_colouring(call.args, m_call_state.sections, m_groups,
                                           m_chunk_size, limits, m_wavefront,
                                           options().window)
                    : plan_chunk_atomics(call.args, m_call_state.sections, m_groups,
                                         m_chunk_size, limits, m_unstaged_singles,
                                         chunk_options.min_compression);
            report(result);
            auto entry = std::make_shared<ChunkEntry>();
            entry->rejected = result.rejected;
            entry->plan = std::move(result.plan);
            if (entry->rejected.empty()) {
                entry->device = DevicePlan(*entry->plan);
                entry->plan->release_uploaded();
            }
            return entry;
        });
    });
}

inline void ChunkBase::report(const ChunkResult& result) const {
    if (OP_diags <= 3)
        return;
    if (result.plan == nullptr) {
        op_printf("%s: %s rejected (%s, compression %.2fx)\n", m_name.c_str(),
                    m_loop.name().c_str(), std::string{result.rejected}.c_str(),
                    result.compression);
        return;
    }
    const auto& plan = *result.plan;
    if (m_coloured)
        op_printf("%s: %s chunk %d, %d chunks, %d batches, up to %d thread "
                    "colours, compression %.2fx\n",
                    m_name.c_str(), m_loop.name().c_str(), plan.chunks.size,
                    plan.chunks.count(), plan.batches.count(),
                    plan.max_thread_colours, plan.compression);
    else
        op_printf("%s: %s chunk %d, %d chunks, %zu refs, %zu targets "
                    "(%zu unstaged), compression %.2fx\n",
                    m_name.c_str(), m_loop.name().c_str(), plan.chunks.size,
                    plan.chunks.count(), plan.references, plan.distinct_targets,
                    plan.staging.unstaged_singles, plan.compression);
}

inline op::loop::Prepared ChunkBase::prepare(
    const op::loop::Call& call, op::loop::Urgency urgency,
    op::loop::KernelKind kind) {
    begin(call);
    const auto *entry = &entry_for(call);
    if (!entry->rejected.empty()) {
        op::loop::Prepared prepared;
        prepared.status = op::loop::Status::unavailable;
        prepared.reason = entry->rejected;
        prepared.entry = entry;
        return prepared;
    }

    // Reduction scratch uses the capped physical grid.  A wavefront batch's
    // blocks take chunks in turn, so it needs no more of them than the
    // device holds at once.
    const auto& batches = entry->plan->batches;
    for (int section = 0; section < m_call_state.sections.size(); ++section)
        for (int batch = 0; batch < batches.batches(section); ++batch) {
            auto [begin, end] = batches.positions(section, batch);
            m_call_state.max_blocks =
                std::max(m_call_state.max_blocks, batch_blocks(*entry->plan, begin, end));
        }
    return acquire(call, urgency, kind, entry);
}

// A block per chunk up to the call's limit, and for a wavefront no more
// than the device holds at once.
inline int ChunkBase::batch_blocks(const ChunkPlan& plan, int begin, int end) const {
    int blocks = std::min(end - begin, m_call_state.block_limit);
    if (plan.wavefront)
        blocks = std::min(blocks, device().resident_blocks(m_block_size));
    return blocks;
}

inline std::optional<float> ChunkBase::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = m_call_state.sections;

    // Every section of a wavefront call takes its positions from zero.
    int epoch = 0;
    if (plan.wavefront) {
        epoch = entry.device.next_wave_epoch();
        GPU_CHECK(gpuMemsetAsync(entry.device.wave_tickets, 0,
                                 static_cast<std::size_t>(sections.size()) *
                                     sizeof(int),
                                 0));
    }

    return run_phases(
        prepared, launch, sections.size(), sections.wait_index(),
        sections.globals_index(),
        [&](int section) { return plan.batches.batches(section); },
        [&](int section, int batch, LaunchContext& context) {
            auto [begin, end] = plan.batches.positions(section, batch);
            context.start = sections[section].begin;
            context.end = sections[section].end;
            context.chunk.plan = &entry.device;
            context.chunk.chunk_begin = begin;
            context.chunk.chunk_end = end;
            if (plan.wavefront) {
                context.chunk.wave_ticket = entry.device.wave_tickets + section;
                context.chunk.wave_epoch = epoch;
            }
            auto bytes = plan.scratch_bytes(section, batch);
            assert(bytes <= static_cast<std::size_t>(INT32_MAX));
            return Geometry{batch_blocks(plan, begin, end), static_cast<int>(bytes)};
        });
}

inline std::string ChunkBase::statistics(const op::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    if (entry.plan == nullptr)
        return std::string{entry.rejected};
    const auto& plan = *entry.plan;
    char text[96];
    std::snprintf(text, sizeof(text), ", chunk %d, %d chunks, compression %.2f",
                  plan.chunks.size, plan.chunks.count(), plan.compression);
    std::string out = StrategyBase::statistics(prepared) + text;
    if (m_coloured)
        out += ", " + std::to_string(plan.batches.count()) + " batches, " +
               std::to_string(plan.max_thread_colours) + " thread colours";
    return out;
}

inline void ChunkBase::release() {
    StrategyBase::release();
    m_plans.clear();
}

} // namespace op::gpu
