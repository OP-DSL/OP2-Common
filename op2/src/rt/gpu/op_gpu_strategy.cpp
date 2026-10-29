#include <op2/rt/gpu/strategy.h>

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/plan/options.h>
#include <op_profile.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace op::rt::gpu {

namespace {

// The live plans of each kind, which every strategy that would build the
// same one shares.
op::rt::plan::SharedPlans<ColouringPlan> colouring_plans;
op::rt::plan::SharedPlans<ChunkEntry> chunk_plans;

} // namespace

op::rt::loop::Axis block_size_axis() {
#ifdef OP2_HIP
    return {"BLOCK_SIZES", "block_size", false, {default_block_size, 256}};
#else
    return {"BLOCK_SIZES", "block_size", false, {default_block_size, 64, 256}};
#endif
}

ColouringPlan::~ColouringPlan() {
    if (order_d != nullptr)
        GPU_CHECK(gpuFree(order_d));
}

// --- The base of every GPU strategy ------------------------------------------------

StrategyBase::StrategyBase(Loop& loop, const op::rt::loop::Point& point, KernelSource source,
                           Binder bind)
    : m_loop{loop}, m_name{point.name}, m_source{source}, m_bind{bind},
      m_block_size{point.value("block_size")}, m_offline{source.offline},
      m_cache{source.name, options().build_threads} {}

op::rt::loop::Estimate StrategyBase::estimate(op::rt::loop::KernelKind kind) const {
    if (kind == op::rt::loop::KernelKind::offline)
        return offline_usable() ? op::rt::loop::Estimate{true, 1.0f}
                                : op::rt::loop::Estimate{false, {}, "kernel_unavailable"};
    if (m_source.source.empty())
        return {false, {}, "kernel_unavailable"};
    bool prior = m_offline.registers() > options().min_registers;
    return {true, prior ? 0.9f : 1.0f};
}

std::string StrategyBase::statistics(const op::rt::loop::Prepared& prepared) const {
    if (prepared.kernel != nullptr)
        return "jit, " + static_cast<JitKernel *>(prepared.kernel)->figures().text();
    return "offline, " + m_offline.figures().text();
}

void StrategyBase::begin(const op::rt::loop::Call& call) {
    m_call_state = {};
    m_call_state.sections = m_loop.sections(call);
    m_call_state.block_limit =
        m_loop.block_limit(const_cast<op_arg *>(call.args.data()),
                           static_cast<int>(call.args.size()), m_block_size);
}

op::rt::loop::Prepared StrategyBase::acquire(const op::rt::loop::Call& call,
                                             op::rt::loop::Urgency urgency,
                                             op::rt::loop::KernelKind kind,
                                             const void *entry) {
    m_call_state.global_stride = m_block_size * m_call_state.max_blocks;
    op::rt::loop::Prepared prepared;
    prepared.entry = entry;

    if (kind == op::rt::loop::KernelKind::offline) {
        prepared.status = op::rt::loop::Status::ready;
        return prepared;
    }

    auto& params = m_loop.params();
    auto key = params.key(call, launch_values().slots());
    auto acquired =
        m_cache.acquire(key, urgency, [&] { return m_loop.compiler().job(params, m_source); });
    // A kernel without launch bounds may have too many registers for its block.
    if (acquired.status == op::rt::loop::Status::ready &&
        acquired.kernel->figures().max_threads < m_block_size) {
        prepared.reason = "thread_limit";
        return prepared;
    }
    prepared.status = acquired.status;
    prepared.kernel = acquired.kernel;
    prepared.reason = acquired.reason;
    return prepared;
}

void StrategyBase::launch(const op::rt::loop::Prepared& prepared, LaunchContext& context,
                          op_arg *args, Geometry geometry) {
    auto& arguments = m_loop.arguments();
    m_bind(context, args, arguments);
    if (prepared.kernel != nullptr) {
        op_profile_next("JIT Kernel");
        static_cast<JitKernel *>(prepared.kernel)
            ->launch(geometry.blocks, m_block_size, arguments.data(), geometry.shared_bytes,
                     m_loop.params());
        return;
    }

    op_profile_next("Offline Kernel");
    m_offline.launch(geometry.blocks, m_block_size, arguments.data(), geometry.shared_bytes);
}

template<typename Launches, typename Fill>
std::optional<float> StrategyBase::run_phases(const op::rt::loop::Prepared& prepared,
                                              const op::rt::loop::Launch& launch, int phases,
                                              int wait_before, int globals_after,
                                              Launches&& launches, Fill&& fill) {
    assert(prepared.candidate->strategy == this);
    // An offline kernel reads the call's and this strategy's values from
    // device symbols.
    if (prepared.kernel == nullptr)
        m_loop.upload(launch_values());
    int timed_pairs = 0;
    m_loop.run_phases(launch, phases, wait_before, globals_after, [&](int phase) {
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
    });

    if (!launch.timed)
        return std::nullopt;
    if (timed_pairs == 0)
        return 0.0f;
    GPU_CHECK(gpuEventSynchronize(m_loop.event(2 * timed_pairs - 1)));
    float total = 0.0f;
    for (int pair = 0; pair < timed_pairs; ++pair) {
        float ms = 0.0f;
        GPU_CHECK(gpuEventElapsedTime(&ms, m_loop.event(2 * pair), m_loop.event(2 * pair + 1)));
        total += ms;
    }
    return total * 1000.0f;
}

// --- Plain and atomics ---

op::rt::loop::Prepared Simple::prepare(const op::rt::loop::Call& call,
                                       op::rt::loop::Urgency urgency,
                                       op::rt::loop::KernelKind kind) {
    begin(call);
    for (const auto& section : m_call_state.sections.span())
        m_call_state.max_blocks = std::max(m_call_state.max_blocks, grid(section.size()));
    return acquire(call, urgency, kind, nullptr);
}

std::optional<float> Simple::run(const op::rt::loop::Prepared& prepared,
                                 const op::rt::loop::Launch& launch) {
    const auto& sections = m_call_state.sections;
    return run_phases(
        prepared, launch, sections.size(), sections.wait_index(), sections.globals_index(),
        [&](int phase) { return sections[phase].size() > 0 ? 1 : 0; },
        [&](int phase, int, LaunchContext& context) {
            context.start = sections[phase].begin;
            context.end = sections[phase].end;
            return Geometry{grid(sections[phase].size()), 0};
        });
}

// --- Colouring ---

op::rt::loop::Prepared Colouring::prepare(const op::rt::loop::Call& call,
                                          op::rt::loop::Urgency urgency,
                                          op::rt::loop::KernelKind kind) {
    begin(call);
    const auto& entry = m_plans.get(call, [&] {
        op::rt::plan::PlanKey key{"colouring", call};
        return colouring_plans.get(key, [&] {
            auto plan = std::make_shared<ColouringPlan>();
            plan->colouring = op::rt::plan::colour_set(call.args, m_call_state.sections);
            const auto& order = plan->colouring.order;
            if (!order.empty()) {
                GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&plan->order_d),
                                    order.size() * sizeof(int)));
                GPU_CHECK(gpuMemcpy(plan->order_d, order.data(), order.size() * sizeof(int),
                                    gpuMemcpyHostToDevice));
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
        m_call_state.max_blocks =
            std::max(m_call_state.max_blocks,
                     grid(colouring.offsets[static_cast<std::size_t>(colour) + 1] -
                          colouring.offsets[static_cast<std::size_t>(colour)]));
    return acquire(call, urgency, kind, &entry);
}

std::optional<float> Colouring::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Launch& launch) {
    const auto& entry = *static_cast<const ColouringPlan *>(prepared.entry);
    const auto& colouring = entry.colouring;
    // The core's colours come first, and run while the halo arrives.
    return run_phases(
        prepared, launch, colouring.colours, colouring.core_colours,
        colouring.owned_colours - 1, [](int) { return 1; },
        [&](int colour, int, LaunchContext& context) {
            context.start = colouring.offsets[static_cast<std::size_t>(colour)];
            context.end = colouring.offsets[static_cast<std::size_t>(colour) + 1];
            context.colour_order = entry.order_d;
            return Geometry{grid(context.end - context.start), 0};
        });
}

std::string Colouring::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const ColouringPlan *>(prepared.entry);
    return StrategyBase::statistics(prepared) + ", " +
           std::to_string(entry.colouring.colours) + " colours";
}

void Colouring::release() {
    StrategyBase::release();
    m_plans.clear();
}

// --- The chunked strategies ---

std::vector<op::rt::loop::Axis> ChunkAtomics::axes() {
#ifdef OP2_HIP
    return chunk_axes({"UNSTAGED_SINGLES", "singles", true, {1, 0}});
#else
    return chunk_axes({"UNSTAGED_SINGLES", "singles", true, {0}});
#endif
}

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, groups, block size and variant; a
// chunk is a block's elements.
const ChunkEntry& ChunkBase::entry_for(const op::rt::loop::Call& call) {
    return m_plans.get(call, [&] {
        ChunkLimits limits{m_block_size, device().shared_bytes};
        const auto& chunk_options = op::rt::plan::chunk_options();
        op::rt::plan::PlanKey key{m_coloured ? "chunk_colouring" : "chunk_atomics", call};
        key.add(m_groups).add(limits.block_size).add(limits.shared_bytes);
        if (m_coloured)
            key.add(m_wavefront).add(options().window);
        else
            key.add(m_unstaged_singles).add(chunk_options.min_compression);
        return chunk_plans.get(key, [&] {
            auto result = m_coloured
                              ? plan_chunk_colouring(call.args, m_call_state.sections,
                                                     m_groups, limits, m_wavefront,
                                                     options().window)
                              : plan_chunk_atomics(call.args, m_call_state.sections,
                                                   m_groups, limits, m_unstaged_singles,
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

void ChunkBase::report(const ChunkResult& result) const {
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
                  plan.chunks.count(), plan.batches.count(), plan.max_thread_colours,
                  plan.compression);
    else
        op_printf("%s: %s chunk %d, %d chunks, %zu refs, %zu targets "
                  "(%zu unstaged), compression %.2fx\n",
                  m_name.c_str(), m_loop.name().c_str(), plan.chunks.size,
                  plan.chunks.count(), plan.references, plan.distinct_targets,
                  plan.staging.unstaged_singles, plan.compression);
}

op::rt::loop::Prepared ChunkBase::prepare(const op::rt::loop::Call& call,
                                          op::rt::loop::Urgency urgency,
                                          op::rt::loop::KernelKind kind) {
    begin(call);
    const auto *entry = &entry_for(call);
    if (!entry->rejected.empty()) {
        op::rt::loop::Prepared prepared;
        prepared.status = op::rt::loop::Status::unavailable;
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
int ChunkBase::batch_blocks(const ChunkPlan& plan, int begin, int end) const {
    int blocks = std::min(end - begin, m_call_state.block_limit);
    if (plan.wavefront)
        blocks = std::min(blocks, device().resident_blocks(m_block_size));
    return blocks;
}

std::optional<float> ChunkBase::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Launch& launch) {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = m_call_state.sections;

    // Every section of a wavefront call takes its positions from zero.
    int epoch = 0;
    if (plan.wavefront) {
        epoch = entry.device.next_wave_epoch();
        GPU_CHECK(gpuMemsetAsync(entry.device.wave_tickets, 0,
                                 static_cast<std::size_t>(sections.size()) * sizeof(int), 0));
    }

    return run_phases(
        prepared, launch, sections.size(), sections.wait_index(), sections.globals_index(),
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

std::string ChunkBase::statistics(const op::rt::loop::Prepared& prepared) const {
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

void ChunkBase::release() {
    StrategyBase::release();
    m_plans.clear();
}

// --- Families ---

std::unique_ptr<op::rt::loop::Strategy> make_strategy(Loop& loop, std::string_view family,
                                                      const op::rt::loop::Point& point,
                                                      const KernelSource& source, Binder bind,
                                                      op::rt::plan::ArgGroups groups) {
    if (family == "plain" || family == "atomics")
        return std::make_unique<Simple>(loop, point, source, bind);
    if (family == "colouring")
        return std::make_unique<Colouring>(loop, point, source, bind);
    if (family == "chunk_atomics")
        return std::make_unique<ChunkAtomics>(loop, point, source, bind, groups);
    if (family == "chunk_colouring")
        return std::make_unique<ChunkColouring>(loop, point, source, bind, groups);
    std::fprintf(stderr, "error: %s registers family %s, which the GPU backend lacks\n",
                 loop.name().c_str(), std::string{family}.c_str());
    std::exit(1);
}

} // namespace op::rt::gpu
