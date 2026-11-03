#include <op2/rt/gpu/strategy.h>

#include <op2/rt/gpu/heuristics.h>
#include <op2/rt/gpu/upload.h>

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
    return {"BLOCK_SIZES", "block_size", false, {abi::gpu::default_block_size, 256}};
#else
    return {"BLOCK_SIZES", "block_size", false, {abi::gpu::default_block_size, 64, 256}};
#endif
}

ColouringPlan::~ColouringPlan() { release(order_d); }

// --- The base of every GPU strategy ------------------------------------------------

StrategyBase::StrategyBase(Loop& loop, const op::rt::loop::Point& point,
                           const abi::gpu::Family& family)
    : m_loop{loop}, m_name{point.name}, m_family{family},
      m_block_size{point.value("block_size")}, m_offline{family.offline, loop.calls()},
      m_cache{family.kernel, options().build_threads} {}

op::rt::loop::Estimate StrategyBase::estimate(op::rt::loop::KernelKind kind) const {
    KernelFacts facts;
    facts.offline = offline_usable();
    facts.resources = m_offline.resources();
    facts.jit = m_family.source != nullptr && m_family.source[0] != '\0';
    return gpu::estimate(facts, kind);
}

std::string StrategyBase::statistics(const op::rt::loop::Prepared& prepared) const {
    if (prepared.kernel != nullptr)
        return "jit, " + describe(static_cast<JitKernel *>(prepared.kernel)->resources());
    return "offline, " + describe(m_offline.resources());
}

void StrategyBase::begin(const op::rt::loop::Call& call) {
    m_call_state = {};
    m_call_state.stride = element_stride(call.set, m_loop.direct());
    m_call_state.block_limit = m_loop.block_limit(call.args, m_block_size);
}

op::rt::loop::Prepared StrategyBase::acquire(const op::rt::loop::Call& call,
                                             op::rt::loop::Urgency urgency,
                                             op::rt::loop::KernelKind kind,
                                             const void *entry) {
    m_call_state.global_stride = m_block_size * m_call_state.max_blocks;
    if (kind == op::rt::loop::KernelKind::offline)
        return op::rt::loop::Prepared::ready(entry);
    op::rt::loop::Prepared prepared;
    prepared.entry = entry;

    auto& params = m_loop.params();
    auto values = launch_values().slots();
    auto key = params.key(call, values);
    auto acquired = m_cache.acquire(
        key, urgency, [&] { return m_loop.compiler().job(params, values, m_family); });
    // A kernel without launch bounds may have too many registers for its block.
    if (acquired.status == op::rt::loop::Status::ready &&
        acquired.kernel->resources().max_threads < m_block_size) {
        prepared.reason = "thread_limit";
        return prepared;
    }
    prepared.status = acquired.status;
    prepared.kernel = acquired.kernel;
    prepared.reason = acquired.reason;
    return prepared;
}

template<typename Record>
void StrategyBase::launch(const op::rt::loop::Prepared& prepared, Record& record,
                          std::span<op_arg> args, Geometry geometry) {
    void **arguments = m_loop.arguments(record, args);
    if (prepared.kernel != nullptr) {
        op_profile_next("JIT Kernel");
        static_cast<JitKernel *>(prepared.kernel)
            ->launch(geometry.blocks, m_block_size, arguments, geometry.shared_bytes,
                     m_loop.params());
        return;
    }

    op_profile_next("Offline Kernel");
    m_offline.launch(geometry.blocks, m_block_size, arguments, geometry.shared_bytes);
}

template<typename Record, typename Launches, typename Fill>
std::optional<float> StrategyBase::run_phases(const op::rt::loop::Prepared& prepared,
                                              const op::rt::loop::RunArgs& run,
                                              const op::rt::loop::Phases& phases,
                                              Launches&& launches, Fill&& fill) {
    assert(prepared.candidate->strategy == this);
    // An offline kernel reads the call's constants and strides from device
    // symbols.
    if (prepared.kernel == nullptr)
        m_loop.upload();
    int timed_pairs = 0;
    m_loop.run_phases(run, phases, [&](int phase) {
        int count = launches(phase);
        bool timing = run.timed && count > 0;
        if (timing)
            GPU_CHECK(gpuEventRecord(m_loop.event(2 * timed_pairs), 0));

        for (int index = 0; index < count; ++index) {
            Record record;
            record.stride = m_call_state.stride;
            record.global_stride = m_call_state.global_stride;
            Geometry geometry = fill(phase, index, record);
            if (geometry.blocks > 0)
                launch(prepared, record, run.call.args, geometry);
        }

        if (timing)
            GPU_CHECK(gpuEventRecord(m_loop.event(2 * timed_pairs++ + 1), 0));
    });

    if (!run.timed)
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
    for (const auto& section : call.sections.span())
        m_call_state.max_blocks = std::max(m_call_state.max_blocks, grid(section.size()));
    return acquire(call, urgency, kind, nullptr);
}

std::optional<float> Simple::run(const op::rt::loop::Prepared& prepared,
                                 const op::rt::loop::RunArgs& run) {
    const auto& sections = run.call.sections;
    return run_phases<abi::gpu::Range>(
        prepared, run, sections.phases(),
        [&](int phase) { return sections[phase].size() > 0 ? 1 : 0; },
        [&](int phase, int, abi::gpu::Range& record) {
            record.start = sections[phase].begin;
            record.end = sections[phase].end;
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
            plan->colouring = op::rt::plan::colour_set(call.args, call.sections);
            upload(plan->colouring.order, plan->order_d);
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
                                    const op::rt::loop::RunArgs& run) {
    const auto& entry = *static_cast<const ColouringPlan *>(prepared.entry);
    const auto& colouring = entry.colouring;
    // The core's colours come first, and run while the halo arrives.
    return run_phases<abi::gpu::Range>(
        prepared, run, colouring.phases(), [](int) { return 1; },
        [&](int colour, int, abi::gpu::Range& record) {
            record.start = colouring.offsets[static_cast<std::size_t>(colour)];
            record.end = colouring.offsets[static_cast<std::size_t>(colour) + 1];
            record.order = entry.order_d;
            return Geometry{grid(record.end - record.start), 0};
        });
}

std::string Colouring::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const ColouringPlan *>(prepared.entry);
    return StrategyBase::statistics(prepared) + ", " +
           std::to_string(entry.colouring.colours) + " colours";
}

// --- The chunked strategies ---

ChunkResult ChunkAtomics::make_plan(const op::rt::loop::Call& call,
                                    const ChunkSpec& spec) const {
    return plan_chunk_atomics(call.args, call.sections, m_groups, spec, m_unstaged_singles,
                              op::rt::plan::chunk_options().min_compression);
}

void ChunkAtomics::add_key(op::rt::plan::PlanKey& key) const {
    key.add(m_unstaged_singles).add(op::rt::plan::chunk_options().min_compression);
}

std::string ChunkAtomics::figures(const ChunkPlan& plan) const {
    char text[96];
    std::snprintf(text, sizeof(text), "%zu refs, %zu targets (%zu unstaged)", plan.references,
                  plan.distinct_targets, plan.staging.unstaged_singles);
    return text;
}

ChunkResult ChunkColouring::make_plan(const op::rt::loop::Call& call,
                                      const ChunkSpec& spec) const {
    return plan_chunk_colouring(call.args, call.sections, m_groups, spec, m_wavefront,
                                options().window);
}

void ChunkColouring::add_key(op::rt::plan::PlanKey& key) const {
    key.add(m_wavefront).add(options().window);
}

std::string ChunkColouring::figures(const ChunkPlan& plan) const {
    return std::to_string(plan.batches.count()) + " batches, up to " +
           std::to_string(plan.max_thread_colours) + " thread colours";
}

std::vector<op::rt::loop::Axis> ChunkAtomics::axes() {
#ifdef OP2_HIP
    return chunk_axes({"UNSTAGED_SINGLES", "singles", true, {1, 0}});
#else
    return chunk_axes({"UNSTAGED_SINGLES", "singles", true, {0}});
#endif
}

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, groups, block size, stride and
// variant; a chunk is a block's elements.
const ChunkEntry& ChunkBase::entry_for(const op::rt::loop::Call& call) {
    return m_plans.get(call, [&] {
        ChunkSpec spec{m_block_size, device().shared_bytes, m_call_state.stride};
        op::rt::plan::PlanKey key{m_family.name, call};
        key.add(m_groups).add(spec.block_size).add(spec.shared_bytes).add(spec.stride);
        add_key(key);
        return chunk_plans.get(key, [&] {
            auto result = make_plan(call, spec);
            report(result);
            auto entry = std::make_shared<ChunkEntry>();
            static_cast<ChunkResult&>(*entry) = std::move(result);
            if (*entry) {
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
    if (!result) {
        op_printf("%s: %s rejected (%s, compression %.2fx)\n", m_name.c_str(),
                  m_loop.name().c_str(), std::string{result.rejected}.c_str(),
                  result.compression);
        return;
    }
    const auto& plan = *result.plan;
    op_printf("%s: %s chunk %d, %d chunks, %s, compression %.2fx\n", m_name.c_str(),
              m_loop.name().c_str(), plan.chunks.size, plan.chunks.count(),
              figures(plan).c_str(), plan.compression);
}

op::rt::loop::Prepared ChunkBase::prepare(const op::rt::loop::Call& call,
                                          op::rt::loop::Urgency urgency,
                                          op::rt::loop::KernelKind kind) {
    begin(call);
    const auto *entry = &entry_for(call);
    if (!*entry)
        return op::rt::loop::Prepared::unavailable(entry->rejected, entry);

    // Reduction scratch uses the capped physical grid.  A wavefront batch's
    // blocks take chunks in turn, so it needs no more of them than the
    // device holds at once.
    const auto& batches = entry->plan->batches;
    for (int section = 0; section < call.sections.size(); ++section)
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
                                    const op::rt::loop::RunArgs& run) {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = run.call.sections;

    // Every section of a wavefront call takes its positions from zero.
    int epoch = 0;
    if (plan.wavefront) {
        epoch = entry.device.next_wave_epoch();
        GPU_CHECK(gpuMemsetAsync(entry.device.wave_tickets, 0,
                                 static_cast<std::size_t>(sections.size()) * sizeof(int), 0));
    }

    return run_phases<abi::gpu::Chunk>(
        prepared, run, sections.phases(),
        [&](int section) { return plan.batches.batches(section); },
        [&](int section, int batch, abi::gpu::Chunk& record) {
            auto [begin, end] = plan.batches.positions(section, batch);
            entry.device.fill(record);
            record.chunk_begin = begin;
            record.chunk_end = end;
            if (plan.wavefront) {
                record.wave_ticket = entry.device.wave_tickets + section;
                record.wave_epoch = epoch;
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
    std::snprintf(text, sizeof(text), ", chunk %d, %d chunks, compression %.2f, ",
                  plan.chunks.size, plan.chunks.count(), plan.compression);
    return StrategyBase::statistics(prepared) + text + figures(plan);
}

// --- Families ---

namespace {

template<typename S>
std::unique_ptr<op::rt::loop::Strategy> make(Loop& loop, const abi::gpu::Family& family,
                                             const op::rt::loop::Point& point) {
    return std::make_unique<S>(loop, point, family);
}

} // namespace

std::span<const StrategyFamily> strategy_families() {
#ifdef OP2_HIP
    constexpr bool chunk_atomics = true;
#else
    constexpr bool chunk_atomics = false;
#endif
    static const std::vector<StrategyFamily> families{
        {{"plain", Simple::axes()}, &make<Simple>},
        {{"chunk_atomics", ChunkAtomics::axes(), chunk_atomics}, &make<ChunkAtomics>},
        {{"atomics", Simple::axes()}, &make<Simple>},
        {{"chunk_colouring", ChunkColouring::axes(), false}, &make<ChunkColouring>},
        {{"colouring", Colouring::axes()}, &make<Colouring>},
    };
    return families;
}

const StrategyFamily& strategy_family(std::string_view name, std::string_view loop) {
    return op::rt::loop::offered_family(strategy_families(), name, loop, "GPU");
}

} // namespace op::rt::gpu
