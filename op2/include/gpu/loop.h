#pragma once

// The CUDA and HIP backend's loop: the object the generated code builds per
// par_loop, from plain data, holding the ladder, the tuner and the JIT
// service, and running one call through them.

#include <gpu/jit.h>
#include <gpu/strategy.h>
#include <loop/autotune.h>
#include <loop/strategy.h>
#include <op_lib_cpp.h>
#include <op_profile.h>
#include <plan/options.h>

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::gpu {

class Loop {
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

    static bool is_type(const char *actual, const char *expected) {
        return actual != nullptr && std::strcmp(actual, expected) == 0;
    }

    void reduce_mpi_globals(op_arg *args, int nargs) {
        for (int i = 0; i < nargs; ++i) {
            auto& arg = args[i];
            if (arg.opt == 0 || arg.argtype != OP_ARG_GBL ||
                (arg.acc != OP_INC && arg.acc != OP_MIN && arg.acc != OP_MAX))
                continue;

            if (is_type(arg.type, "double") || is_type(arg.type, "r8") ||
                is_type(arg.type, "real*8") || is_type(arg.type, "real(8)")) {
                op_mpi_reduce_double(&arg, reinterpret_cast<double *>(arg.data));
            } else if (is_type(arg.type, "float") || is_type(arg.type, "r4") ||
                       is_type(arg.type, "real*4") || is_type(arg.type, "real(4)")) {
                op_mpi_reduce_float(&arg, reinterpret_cast<float *>(arg.data));
            } else if (is_type(arg.type, "int") || is_type(arg.type, "i4") ||
                       is_type(arg.type, "integer*4") ||
                       is_type(arg.type, "integer(4)")) {
                op_mpi_reduce_int(&arg, reinterpret_cast<int *>(arg.data));
            } else if (is_type(arg.type, "bool") || is_type(arg.type, "logical")) {
                op_mpi_reduce_bool(&arg, reinterpret_cast<bool *>(arg.data));
            } else {
                std::fprintf(stderr,
                             "error: unsupported MPI reduction type '%s' (in %s)\n",
                             arg.type == nullptr ? "<null>" : arg.type,
                             m_name.c_str());
                std::exit(1);
            }
        }
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
    const LoopDescription& description() const { return m_description; }
    op::loop::Ladder& ladder() { return m_ladder; }

    op::loop::Sections sections(const op::loop::Configuration& configuration) const {
        if (m_description.direct)
            return op::loop::Sections::direct(configuration.set);
        return op::loop::Sections::indirect(configuration.set,
                                            configuration.global_reduction());
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

    // Run the loop: ready once it ran, else the floor's status, with nothing
    // run and no exchange started.
    op::loop::Status invoke(op_set set, op_arg *args, int nargs) {
        op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                                m_profile_variant.c_str());
        op_profile_enter("Init");
        op_profile_enter("Get Kernel");

        m_jit.update(args, nargs);
        op::loop::Configuration configuration{set, {args, static_cast<std::size_t>(nargs)}};
        auto selection = m_ladder.select(configuration);
        if (selection.prepared.status != op::loop::Status::ready) {
            op_profile_exit();
            op_profile_exit();
            op_profile_exit();
            return selection.prepared.status;
        }
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
            reduce_mpi_globals(args, nargs);
            op_mpi_set_dirtybit_cuda(nargs, args);
            op_profile_exit();
            return op::loop::Status::ready;
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
        reduce_mpi_globals(args, nargs);
        op_mpi_set_dirtybit_cuda(nargs, args);
        op_profile_exit();
        op_profile_exit();
        return op::loop::Status::ready;
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

inline void StrategyBase::begin(const op::loop::Configuration& configuration) {
    m_call = {};
    m_call.sections = m_loop.sections(configuration);
    m_call.block_limit = m_loop.block_limit(
        const_cast<op_arg *>(configuration.args.data()),
        static_cast<int>(configuration.args.size()), m_block_size);
}

inline op::loop::Prepared StrategyBase::acquire(bool now, const void *entry) {
    m_call.global_stride = m_block_size * m_call.max_blocks;
    m_loop.jit().set_launch(
        {m_call.global_stride, m_block_size, m_unstaged_singles, m_wavefront});
    auto acquired = m_loop.jit().acquire(m_kernels, m_source, m_block_size, now);
    m_call.kernel = acquired.kernel;

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
    m_loop.jit().upload();
    m_kernels.launch_offline(m_source.offline, geometry.blocks, m_block_size,
                             offline.data(), geometry.shared_bytes);
}

template<typename Launches, typename Fill>
std::optional<float> StrategyBase::run_phases(
    const op::loop::Prepared& prepared, const op::loop::Launch& launch,
    int phases, int wait_before, int globals_after, Launches&& launches,
    Fill&& fill) {
    assert(prepared.strategy == this);
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
    int wait = 0;
    int globals = 0;
    for (int i = 0; i <= sections.size(); ++i) {
        if (sections.wait_before(i))
            wait = i;
        if (i < sections.size() && sections.process_globals_after(i))
            globals = i;
    }
    return run_phases(
        prepared, launch, sections.size(), wait, globals,
        [&](int phase) { return sections[phase].size() > 0 ? 1 : 0; },
        [&](int phase, int, LaunchContext& context) {
            context.start = sections[phase].begin;
            context.end = sections[phase].end;
            return Geometry{grid(sections[phase].size()), 0};
        });
}

// --- Colouring ---

inline op::loop::Prepared Colouring::prepare(
    const op::loop::Configuration& configuration, bool now) {
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
            entry.order_d = static_cast<int *>(
                op::device().malloc(order.size() * sizeof(int)));
            op::device().copy(entry.order_d, order.data(), order.size() * sizeof(int));
        }
        if (OP_diags > 3)
            std::printf("colouring: %s %d colours, %d over owned elements\n",
                        m_loop.name().c_str(), entry.colouring.colours,
                        entry.colouring.owned_colours);
        found = m_entries.emplace(key, std::move(entry)).first;
    }
    m_entry = &found->second;

    const auto& colouring = m_entry->colouring;
    for (int colour = 0; colour < colouring.colours; ++colour)
        m_call.max_blocks = std::max(
            m_call.max_blocks,
            grid(colouring.offsets[static_cast<std::size_t>(colour) + 1] -
                 colouring.offsets[static_cast<std::size_t>(colour)]));
    return acquire(now, m_entry);
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
            op::device().free(entry.order_d);
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
    Entry entry;
    entry.rejected = result.rejected;
    entry.plan = std::move(result.plan);
    if (entry.rejected.empty())
        entry.device = DevicePlan(*entry.plan);
    report(entry);
    return m_entries.emplace(key, std::move(entry)).first->second;
}

inline void ChunkBase::report(const Entry& entry) const {
    if (OP_diags <= 3)
        return;
    if (entry.plan == nullptr) {
        std::printf("%s: %s rejected (%s)\n", m_name.c_str(),
                    m_loop.name().c_str(), std::string{entry.rejected}.c_str());
        return;
    }
    const auto& plan = *entry.plan;
    if (m_coloured)
        std::printf("%s: %s chunk %d, %d chunks, %d batches, up to %d thread "
                    "colours, compression %.2fx\n",
                    m_name.c_str(), m_loop.name().c_str(), plan.chunks.size,
                    plan.chunks.count(), plan.batches.count(),
                    plan.max_thread_colours, plan.compression());
    else
        std::printf("%s: %s chunk %d, %d chunks, %zu refs, %zu targets "
                    "(%zu unstaged), compression %.2fx%s\n",
                    m_name.c_str(), m_loop.name().c_str(), plan.chunks.size,
                    plan.chunks.count(), plan.references, plan.distinct_targets,
                    plan.staging.unstaged_singles, plan.compression(),
                    entry.rejected.empty() ? "" : ", rejected");
}

inline op::loop::Prepared ChunkBase::prepare(
    const op::loop::Configuration& configuration, bool now) {
    if (never_available())
        return unavailable();
    begin(configuration);
    m_entry = &entry_for(configuration);
    if (!m_entry->rejected.empty()) {
        op::loop::Prepared prepared;
        prepared.strategy = this;
        prepared.status = op::loop::Status::rejected;
        prepared.reason = m_entry->rejected;
        prepared.entry = m_entry;
        return prepared;
    }

    // Reduction scratch uses the capped physical grid.  A wavefront batch's
    // blocks take chunks in turn, so it needs no more of them than the
    // device holds at once.
    const auto& batches = m_entry->plan->batches;
    for (int section = 0; section < m_call.sections.size(); ++section)
        for (int batch = 0; batch < batches.batches(section); ++batch) {
            auto [begin, end] = batches.positions(section, batch);
            int blocks = std::min(end - begin, m_call.block_limit);
            if (m_entry->plan->wavefront)
                blocks = std::min(blocks, config().resident_blocks(m_block_size));
            m_call.max_blocks = std::max(m_call.max_blocks, blocks);
        }
    return acquire(now, m_entry);
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

    int wait = 0;
    int globals = 0;
    for (int i = 0; i <= sections.size(); ++i) {
        if (sections.wait_before(i))
            wait = i;
        if (i < sections.size() && sections.process_globals_after(i))
            globals = i;
    }
    return run_phases(
        prepared, launch, sections.size(), wait, globals,
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
    m_entry = nullptr;
}

} // namespace op::gpu
