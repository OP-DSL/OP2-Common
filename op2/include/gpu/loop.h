#pragma once

// The CUDA and HIP backend's loop: the object the generated code builds per
// par_loop, from plain data, holding the ladder, the tuner and the JIT
// service, and running one call through them.

#include <gpu/fusion.h>
#include <gpu/jit.h>
#include <gpu/lazy.h>
#include <gpu/strategy.h>
#include <loop/autotune.h>
#include <loop/strategy.h>
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

class Loop : public QueuedLoop {
private:
    std::string m_name;
    std::string m_profile_target;
    std::string m_profile_variant;
    LoopDescription m_description;
    op::loop::Ladder m_ladder;
    std::optional<op::loop::Autotuner> m_tuner;
    JitService m_jit;
    GlobalsInit m_init_globals;
    // Event pairs bracketing each phase's launches of a timed call.
    std::vector<gpuEvent_t> m_events;
    // A launch's kernel arguments, offline and JIT, filled by the binders.
    std::vector<void *> m_offline_args;
    std::vector<void *> m_jit_args;
    // The call under way: whether any global is written, and whether its
    // processing needs a synchronisation before the host reads.
    bool m_global_output = false;
    bool m_exit_sync = false;

    static void release_callback(void *owner) {
        static_cast<Loop *>(owner)->release();
    }

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

public:
    Loop(const Loop&) = delete;

    // profile_target and profile_variant name the loop to the profiler;
    // init_globals, if any, initialises the per-thread reduction storage.
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, LoopDescription description,
         JitHeaders headers, GlobalsInit init_globals = nullptr)
        : m_name{name}, m_profile_target{profile_target},
          m_profile_variant{profile_variant}, m_description{description},
          m_ladder{name}, m_jit{name, std::move(headers)},
          m_init_globals{init_globals} {
        if (op::loop::autotune_config().report)
            m_tuner.emplace(m_ladder);
        op::loop::register_release(this, &Loop::release_callback);
        op_lazy_set_flush(&flush_queue);
        queue().set_runner([](std::vector<PendingCall>& calls) { fusion().run(calls); });
    }

    ~Loop() {
        op::loop::unregister_release(this);
        release();
    }

    // --- Registration, by the generated code ---

    // Add a family: a strategy S per point of S's axes, built from the
    // loop, the point and args.
    template<typename S, typename... Args>
    void add(std::string_view family, op::loop::Default default_setting,
             Args&&... args) {
        auto axes = S::axes();
        m_ladder.add(family, default_setting, axes,
                     [&](const op::loop::Point& point) {
                         return std::make_unique<S>(*this, point, args...);
                     });
    }

    JitService& jit() { return m_jit; }

    // --- For the strategies ---

    const std::string& name() const { return m_name; }
    op::loop::Ladder& ladder() { return m_ladder; }

    op::loop::Sections sections(const op::loop::Configuration& configuration) const {
        return op::loop::Sections::of(configuration, m_description.direct);
    }

    // The most blocks a launch may use: as many as the reduction scratch
    // allows.
    int block_limit(op_arg *args, int nargs, int block_size) const {
        return getBlockLimitWithPolicy(args, nargs, block_size, m_name.c_str(),
                                       m_description.gbl_inc_atomic);
    }

    std::vector<void *>& offline_args() { return m_offline_args; }
    std::vector<void *>& jit_args() { return m_jit_args; }

    gpuEvent_t event(std::size_t index) {
        while (m_events.size() <= index) {
            gpuEvent_t event;
            CUDA_SAFE_CALL(gpuEventCreate(&event));
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
                                                   global_stride,
                                                   m_description.gbl_inc_atomic);
        op_profile_next("Kernel");
    }

    // --- Per call ---

    // Call the loop: queue the call under OP_LAZY, else run it.
    void invoke(op_set set, op_arg *args, int nargs) {
        if (!config().lazy) {
            m_jit.update(args, nargs);
            run_call(set, args, nargs);
            return;
        }
        auto call = Queue::capture(*this, set, args, nargs);
        m_jit.update(call.args.data(), nargs);
        call.params = m_jit.save();
        queue().push(std::move(call));
    }

    // Run a queued call with the values it was made with.
    void execute(PendingCall& call) override {
        call.bind_globals();
        m_jit.restore(call.params);
        run_call(call.set, call.args.data(), static_cast<int>(call.args.size()));
    }

    // --- Fusion (gpu/fusion.h) ---

    const char *fusion_member(PendingCall& call, FusionMember& member) override {
        if (m_tuner)
            return "tuning";
        if (!config().enable)
            return "JIT off";
        auto nargs = static_cast<int>(call.args.size());
        auto index = m_ladder.preferred();
        if (!index)
            return "no strategy";
        auto& strategy = static_cast<StrategyBase&>(m_ladder[*index]);
        if (strategy.source().fusion == nullptr)
            return strategy.name().data();  // a std::string's, so terminated

        call.bind_globals();
        m_jit.restore(call.params);
        op::loop::Configuration configuration{call.set, call.args};
        auto geometry = strategy.simple_call(configuration);
        const auto& sections = geometry.sections;
        member.call = &call;
        member.strategy = &strategy;
        member.begin = sections[0].begin;
        member.end = sections[sections.size() - 1].end;
        // A direct call reads no halo.
        member.core = m_description.direct ? member.end : call.set->core_size;
        member.owned = call.set->size;
        member.exec = !m_description.direct && configuration.indirect_write();
        member.set_stride = sections.set_stride();
        member.block_size = strategy.block_size();
        member.block_limit = geometry.block_limit;
        member.max_blocks = geometry.max_blocks;
        member.global_stride = geometry.global_stride;
        auto hash = m_jit.fusion_hash({geometry.global_stride, member.block_size, false, false});
        hash = op::gpu::hash_bytes(m_name.data(), m_name.size(), hash);
        member.key = op::gpu::hash_bytes(strategy.name().data(), strategy.name().size(), hash);
        member.global_output = has_global_output(call.args.data(), nargs);
        for (const auto& arg : call.args)
            member.device_globals |=
                arg.opt != 0 && (arg.argtype == OP_ARG_INFO ||
                                 (arg.argtype == OP_ARG_GBL && !(arg.acc == OP_READ && arg.dim == 1)));
        member.registers = strategy.offline_registers();
        return nullptr;
    }

    std::string fusion_params(const FusionMember& member) override {
        m_jit.restore(member.call->params);
        return m_jit.fusion_source({member.global_stride, member.block_size, false, false});
    }

    const JitHeaders& fusion_headers() const override { return m_jit.headers(); }
    const std::string& fusion_name() const override { return m_name; }

    void fused_begin(FusionMember& member) override {
        auto *args = member.call->args.data();
        auto nargs = static_cast<int>(member.call->args.size());
        prepareDeviceGblsWithPolicy(args, nargs, member.global_stride,
                                    m_description.gbl_inc_atomic);
        if (m_init_globals != nullptr)
            m_init_globals({member.block_size, member.max_blocks, member.global_stride}, args);
    }

    void fused_globals(FusionMember& member) override {
        auto *args = member.call->args.data();
        auto nargs = static_cast<int>(member.call->args.size());
        if (member.global_output &&
            processDeviceGblsWithPolicy(args, nargs, member.global_stride,
                                        member.global_stride, m_description.gbl_inc_atomic))
            CUDA_SAFE_CALL(gpuStreamSynchronize(0));
    }

    void fused_end(FusionMember& member) override {
        auto *args = member.call->args.data();
        auto nargs = static_cast<int>(member.call->args.size());
        op::loop::reduce_mpi_globals(args, nargs, m_name.c_str());
        op_mpi_set_dirtybit_cuda(nargs, args);
    }

    // Run a call whose JIT parameters are current.
    void run_call(op_set set, op_arg *args, int nargs) {
        op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                                m_profile_variant.c_str());
        op_profile_enter("Init");
        op_profile_enter("Get Kernel");

        op::loop::Configuration configuration{set, {args, static_cast<std::size_t>(nargs)}};
        auto selection = m_ladder.select(configuration);
        auto decision = m_tuner ? m_tuner->choose(configuration, selection)
                                : op::loop::Autotuner::Decision{
                                      selection.strategy, selection.prepared,
                                      false, nullptr};
        const auto& call = static_cast<StrategyBase *>(decision.strategy)->call();

        op_profile_next("MPI Exchanges");
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
        op_profile_exit();

        op_profile_enter("Prepare GBLs");
        m_global_output = has_global_output(args, nargs);
        m_exit_sync = false;
        prepareDeviceGblsWithPolicy(args, nargs, call.global_stride,
                                    m_description.gbl_inc_atomic);
        if (m_init_globals != nullptr)
            m_init_globals({static_cast<StrategyBase *>(decision.strategy)->block_size(),
                            call.max_blocks, call.global_stride},
                           args);
        op_profile_exit();

        op_profile_next("Computation");
        op_profile_enter("Kernel");
        op::loop::Launch launch{args, nargs, decision.timed};
        auto microseconds = decision.strategy->run(decision.prepared, launch);
        if (microseconds && m_tuner)
            m_tuner->sample(decision, *microseconds);
        op_profile_exit();
        op_profile_exit();

        op_profile_enter("Finalise");
        if (m_exit_sync)
            CUDA_SAFE_CALL(gpuStreamSynchronize(0));
        op::loop::reduce_mpi_globals(args, nargs, m_name.c_str());
        op_mpi_set_dirtybit_cuda(nargs, args);
        op_profile_exit();
        op_profile_exit();
    }

    // At op_exit, while the device is still up: join the compiles, report
    // the tuner, and free every strategy's device state.
    void release() {
        if (m_tuner)
            m_tuner->flush();
        for (std::size_t i = 0; i < m_ladder.size(); ++i)
            static_cast<StrategyBase&>(m_ladder[i]).release();
        for (auto event : m_events)
            gpuEventDestroy(event);
        m_events.clear();
    }
};

// --- The strategies' definitions that need the loop ---------------------------------

inline StrategyBase::StrategyBase(Loop& loop, const op::loop::Point& point,
                                  KernelSource source, Binder bind)
    : m_loop{loop}, m_name{point.name}, m_source{source}, m_bind{bind},
      m_block_size{point.values[0]} {
    m_kernels.register_offline(source.offline);
}

inline StrategyBase::Call StrategyBase::simple_call(
    const op::loop::Configuration& configuration) const {
    Call call;
    call.sections = m_loop.sections(configuration);
    call.block_limit = m_loop.block_limit(
        const_cast<op_arg *>(configuration.args.data()),
        static_cast<int>(configuration.args.size()), m_block_size);
    for (const auto& section : call.sections.span()) {
        int blocks = (section.size() + m_block_size - 1) / m_block_size;
        call.max_blocks = std::max(call.max_blocks, std::min(blocks, call.block_limit));
    }
    call.global_stride = m_block_size * call.max_blocks;
    return call;
}

inline void StrategyBase::begin(const op::loop::Configuration& configuration) {
    m_call = {};
    m_call.sections = m_loop.sections(configuration);
    m_call.block_limit = m_loop.block_limit(
        const_cast<op_arg *>(configuration.args.data()),
        static_cast<int>(configuration.args.size()), m_block_size);
}

inline op::loop::Prepared StrategyBase::acquire(op::loop::Urgency urgency,
                                                const void *entry) {
    m_call.global_stride = m_block_size * m_call.max_blocks;
    auto acquired =
        m_loop.jit().acquire(m_kernels, m_source, launch_values(), urgency);

    op::loop::Prepared prepared;
    prepared.strategy = this;
    prepared.status = acquired.status;
    prepared.entry = entry;
    prepared.kernel = acquired.kernel;
    if (acquired.status == op::loop::Status::pending)
        prepared.reason = "compiling";
    else if (acquired.status == op::loop::Status::unavailable)
        prepared.reason = "kernel_unavailable";
    return prepared;
}

inline void StrategyBase::launch(const op::loop::Prepared& prepared,
                                 LaunchContext& context, op_arg *args,
                                 Geometry geometry) {
    auto& offline = m_loop.offline_args();
    auto& jit = m_loop.jit_args();
    m_bind(context, args, offline, jit);
    if (prepared.kernel != nullptr) {
        op_profile_next("JIT Kernel");
        static_cast<JitKernel *>(prepared.kernel)
            ->launch(geometry.blocks, m_block_size, jit.data(),
                     geometry.shared_bytes);
        return;
    }

    op_profile_next("Offline Kernel");
    m_kernels.launch_offline(m_source.offline, geometry.blocks, m_block_size,
                             offline.data(), geometry.shared_bytes);
}

template<typename Launches, typename Fill>
std::optional<float> StrategyBase::run_phases(
    const op::loop::Prepared& prepared, const op::loop::Launch& launch,
    int phases, int wait_before, int globals_after, Launches&& launches,
    Fill&& fill) {
    assert(prepared.strategy == this);
    // An offline kernel reads the call's and this strategy's values from
    // device symbols.
    if (prepared.kernel == nullptr)
        m_loop.jit().upload(launch_values());
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
            CUDA_SAFE_CALL(gpuEventRecord(m_loop.event(2 * timed_pairs), 0));

        for (int index = 0; index < count; ++index) {
            LaunchContext context;
            context.global_stride = m_call.global_stride;
            context.set_stride = m_call.sections.set_stride();
            Geometry geometry = fill(phase, index, context);
            if (geometry.blocks > 0)
                this->launch(prepared, context, launch.args, geometry);
        }

        if (timing)
            CUDA_SAFE_CALL(gpuEventRecord(m_loop.event(2 * timed_pairs++ + 1), 0));
        if (phase == globals_after)
            m_loop.process_globals(launch.args, launch.nargs, m_call.global_stride);
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
    CUDA_SAFE_CALL(gpuEventSynchronize(m_loop.event(2 * timed_pairs - 1)));
    float total = 0.0f;
    for (int pair = 0; pair < timed_pairs; ++pair) {
        float ms = 0.0f;
        CUDA_SAFE_CALL(gpuEventElapsedTime(&ms, m_loop.event(2 * pair),
                                           m_loop.event(2 * pair + 1)));
        total += ms;
    }
    return total * 1000.0f;
}

// --- Plain and atomics ---

inline std::optional<float> Simple::run(const op::loop::Prepared& prepared,
                                        const op::loop::Launch& launch) {
    const auto& sections = m_call.sections;
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
    const op::loop::Configuration& configuration, op::loop::Urgency urgency) {
    if (never_available())
        return unavailable();
    begin(configuration);
    auto key = configuration.key();
    auto found = m_entries.find(key);
    if (found == m_entries.end()) {
        Entry entry;
        entry.colouring = op::plan::colour_set(
            configuration.set, configuration.args, m_call.sections.size() == 3);
        const auto& order = entry.colouring.order;
        if (!order.empty()) {
            CUDA_SAFE_CALL(gpuMalloc(reinterpret_cast<void **>(&entry.order_d),
                                     order.size() * sizeof(int)));
            CUDA_SAFE_CALL(gpuMemcpy(entry.order_d, order.data(),
                                     order.size() * sizeof(int), gpuMemcpyHostToDevice));
        }
        if (OP_diags > 3)
            std::printf("colouring: %s %d colours, %d over owned elements\n",
                        m_loop.name().c_str(), entry.colouring.colours,
                        entry.colouring.owned_colours);
        found = m_entries.emplace(key, std::move(entry)).first;
    }
    auto *entry = &found->second;

    const auto& colouring = entry->colouring;
    for (int colour = 0; colour < colouring.colours; ++colour)
        m_call.max_blocks = std::max(
            m_call.max_blocks,
            grid(colouring.offsets[static_cast<std::size_t>(colour) + 1] -
                 colouring.offsets[static_cast<std::size_t>(colour)]));
    return acquire(urgency, entry);
}

inline std::optional<float> Colouring::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
    const auto& colouring = entry.colouring;
    // Every colour can hold halo-dependent elements, as op_plan's OP_COLOR2
    // colouring's could, so the wait comes first.
    return run_phases(
        prepared, launch, colouring.colours, 0, colouring.owned_colours - 1,
        [](int) { return 1; },
        [&](int colour, int, LaunchContext& context) {
            context.start = colouring.offsets[static_cast<std::size_t>(colour)];
            context.end = colouring.offsets[static_cast<std::size_t>(colour) + 1];
            context.colour_order = entry.order_d;
            return Geometry{grid(context.end - context.start), 0};
        });
}

inline std::string Colouring::statistics(const op::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
    return StrategyBase::statistics(prepared) + ", " +
           std::to_string(entry.colouring.colours) + " colours";
}

inline void Colouring::release() {
    StrategyBase::release();
    for (auto& [key, entry] : m_entries)
        if (entry.order_d != nullptr)
            CUDA_SAFE_CALL(gpuFree(entry.order_d));
    m_entries.clear();
}

// --- The chunked strategies ---

inline ChunkBase::Entry& ChunkBase::entry_for(
    const op::loop::Configuration& configuration) {
    auto key = configuration.key();
    auto found = m_entries.find(key);
    if (found != m_entries.end())
        return found->second;

    const auto& run = config();
    ChunkLimits limits{m_block_size, run.shared_bytes};
    auto result =
        m_coloured
            ? plan_chunk_colouring(configuration.args, m_call.sections, m_groups,
                                   m_chunk_size, limits, m_wavefront, run.window)
            : plan_chunk_atomics(configuration.args, m_call.sections, m_groups,
                                 m_chunk_size, limits, m_unstaged_singles,
                                 op::plan::chunk_options().min_compression);
    report(result);
    Entry entry;
    entry.rejected = result.rejected;
    entry.plan = std::move(result.plan);
    if (entry.rejected.empty()) {
        entry.device = DevicePlan(*entry.plan);
        entry.plan->release_uploaded();
    }
    return m_entries.emplace(key, std::move(entry)).first->second;
}

inline void ChunkBase::report(const ChunkResult& result) const {
    if (OP_diags <= 3)
        return;
    if (result.plan == nullptr) {
        std::printf("%s: %s rejected (%s, compression %.2fx)\n", m_name.c_str(),
                    m_loop.name().c_str(), std::string{result.rejected}.c_str(),
                    result.compression);
        return;
    }
    const auto& plan = *result.plan;
    if (m_coloured)
        std::printf("%s: %s chunk %d, %d chunks, %d batches, up to %d thread "
                    "colours, compression %.2fx\n",
                    m_name.c_str(), m_loop.name().c_str(), plan.chunks.size,
                    plan.chunks.count(), plan.batches.count(),
                    plan.max_thread_colours, plan.compression());
    else
        std::printf("%s: %s chunk %d, %d chunks, %zu refs, %zu targets "
                    "(%zu unstaged), compression %.2fx\n",
                    m_name.c_str(), m_loop.name().c_str(), plan.chunks.size,
                    plan.chunks.count(), plan.references, plan.distinct_targets,
                    plan.staging.unstaged_singles, plan.compression());
}

inline op::loop::Prepared ChunkBase::prepare(
    const op::loop::Configuration& configuration, op::loop::Urgency urgency) {
    if (never_available())
        return unavailable();
    begin(configuration);
    auto *entry = &entry_for(configuration);
    if (!entry->rejected.empty()) {
        op::loop::Prepared prepared;
        prepared.strategy = this;
        prepared.status = op::loop::Status::rejected;
        prepared.reason = entry->rejected;
        prepared.entry = entry;
        return prepared;
    }

    // Reduction scratch uses the capped physical grid.  A wavefront batch's
    // blocks take chunks in turn, so it needs no more of them than the
    // device holds at once.
    const auto& batches = entry->plan->batches;
    for (int section = 0; section < m_call.sections.size(); ++section)
        for (int batch = 0; batch < batches.batches(section); ++batch) {
            auto [begin, end] = batches.positions(section, batch);
            int blocks = std::min(end - begin, m_call.block_limit);
            if (entry->plan->wavefront)
                blocks = std::min(blocks, config().resident_blocks(m_block_size));
            m_call.max_blocks = std::max(m_call.max_blocks, blocks);
        }
    return acquire(urgency, entry);
}

inline std::optional<float> ChunkBase::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = m_call.sections;

    // Every section of a wavefront call takes its positions from zero.
    int epoch = 0;
    if (plan.wavefront) {
        epoch = entry.device.next_wave_epoch();
        CUDA_SAFE_CALL(gpuMemsetAsync(entry.device.wave_tickets, 0,
                                      static_cast<std::size_t>(sections.size()) *
                                          sizeof(int),
                                      0));
    }

    return run_phases(
        prepared, launch, sections.size(), sections.wait_index(),
        sections.globals_index(),
        [&](int section) {
            return sections[section].size() > 0 ? plan.batches.batches(section) : 0;
        },
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
            int blocks = std::min(end - begin, m_call.block_limit);
            if (plan.wavefront)
                blocks = std::min(blocks, config().resident_blocks(m_block_size));
            auto bytes = plan.scratch_bytes(section, batch);
            assert(bytes <= static_cast<std::size_t>(INT32_MAX));
            return Geometry{blocks, static_cast<int>(bytes)};
        });
}

inline std::string ChunkBase::statistics(const op::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
    if (entry.plan == nullptr)
        return std::string{entry.rejected};
    const auto& plan = *entry.plan;
    char text[96];
    std::snprintf(text, sizeof(text), ", chunk %d, %d chunks, compression %.2f",
                  plan.chunks.size, plan.chunks.count(), plan.compression());
    std::string out = StrategyBase::statistics(prepared) + text;
    if (m_coloured)
        out += ", " + std::to_string(plan.batches.count()) + " batches, " +
               std::to_string(plan.max_thread_colours) + " thread colours";
    return out;
}

inline void ChunkBase::release() {
    StrategyBase::release();
    m_entries.clear();
}

} // namespace op::gpu
