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
    return {.variable = "BLOCK_SIZES",
            .key = "block_size",
            .defaults = {abi::gpu::default_block_size, 256}};
#else
    return {.variable = "BLOCK_SIZES",
            .key = "block_size",
            .defaults = {abi::gpu::default_block_size, 64, 256}};
#endif
}

ColouringPlan::~ColouringPlan() { release(order_d); }

// --- The base of every GPU strategy ------------------------------------------------

StrategyBase::StrategyBase(Executor& executor, const op::rt::loop::Point& point,
                           const abi::gpu::Family& family)
    : m_executor{executor}, m_name{point.name}, m_family{family},
      m_block_size{point.value("block_size")}, m_offline{family.offline, executor.calls()},
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

StrategyBase::Shape StrategyBase::shape(const op::rt::loop::Call& call,
                                        const void *entry) const {
    Shape shape;
    shape.stride = element_stride(call.set, m_executor.direct());
    shape.block_limit = m_executor.block_limit(call, m_block_size);
    shape.global_stride = m_block_size * max_blocks(call, entry, shape.block_limit);
    return shape;
}

op::rt::loop::Prepared StrategyBase::prepare(const op::rt::loop::Call& call,
                                             op::rt::loop::Urgency urgency,
                                             op::rt::loop::KernelKind kind) {
    auto planned = plan(call, element_stride(call.set, m_executor.direct()));
    if (!planned.rejected.empty())
        return op::rt::loop::Prepared::unavailable(planned.rejected, planned.entry);
    if (kind == op::rt::loop::KernelKind::offline)
        return op::rt::loop::Prepared::ready(planned.entry);
    op::rt::loop::Prepared prepared;
    prepared.entry = planned.entry;

    auto& params = m_executor.params();
    auto values = launch_values(shape(call, planned.entry).global_stride).slots();
    auto key = params.key(call, values);
    auto acquired = m_cache.acquire(
        key, urgency, [&] { return m_executor.compiler().job(params, values, m_family); });
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
    void **arguments = m_executor.arguments(record, args);
    if (prepared.kernel != nullptr) {
        op_profile_next("JIT Kernel");
        static_cast<JitKernel *>(prepared.kernel)
            ->launch(geometry.blocks, m_block_size, arguments, geometry.shared_bytes,
                     m_executor.params());
        return;
    }

    op_profile_next("Offline Kernel");
    m_offline.launch(geometry.blocks, m_block_size, arguments, geometry.shared_bytes);
}

template<typename Record, typename Launches, typename Fill>
std::optional<float> StrategyBase::run_phases(const op::rt::loop::Prepared& prepared,
                                              const op::rt::loop::Call& call, bool timed,
                                              const op::rt::loop::Phases& phases,
                                              const Shape& shape, Launches&& launches,
                                              Fill&& fill) {
    assert(prepared.candidate->strategy == this);
    // An offline kernel reads the call's constants and strides from device
    // symbols.
    if (prepared.kernel == nullptr)
        m_executor.upload();
    int timed_pairs = 0;
    bool sync = false;
    op::rt::loop::run_phases(call, phases, [&](int phase) {
        int count = launches(phase);
        bool timing = timed && count > 0;
        if (timing)
            GPU_CHECK(gpuEventRecord(m_executor.event(2 * timed_pairs), 0));

        for (int index = 0; index < count; ++index) {
            Record record;
            record.stride = shape.stride;
            record.global_stride = shape.global_stride;
            Geometry geometry = fill(phase, index, record);
            if (geometry.blocks > 0)
                launch(prepared, record, call.args, geometry);
        }

        if (timing)
            GPU_CHECK(gpuEventRecord(m_executor.event(2 * timed_pairs++ + 1), 0));
    }, [&] { sync |= m_executor.combine_globals(call, shape.global_stride); });
    // The host reads the combined globals next.
    if (sync)
        GPU_CHECK(gpuStreamSynchronize(0));

    if (!timed)
        return std::nullopt;
    if (timed_pairs == 0)
        return 0.0f;
    GPU_CHECK(gpuEventSynchronize(m_executor.event(2 * timed_pairs - 1)));
    float total = 0.0f;
    for (int pair = 0; pair < timed_pairs; ++pair) {
        float ms = 0.0f;
        GPU_CHECK(gpuEventElapsedTime(&ms, m_executor.event(2 * pair),
                                      m_executor.event(2 * pair + 1)));
        total += ms;
    }
    return total * 1000.0f;
}

// --- Plain and atomics ---

int Simple::max_blocks(const op::rt::loop::Call& call, const void *, int block_limit) const {
    int blocks = 0;
    for (const auto& section : call.sections.span())
        blocks = std::max(blocks, grid(section.size(), block_limit));
    return blocks;
}

std::optional<float> Simple::run(const op::rt::loop::Prepared& prepared,
                                 const op::rt::loop::Call& call, bool timed) {
    const auto& sections = call.sections;
    auto call_shape = shape(call, prepared.entry);
    return run_phases<abi::gpu::Range>(
        prepared, call, timed, sections.phases(), call_shape,
        [&](int phase) { return sections[phase].size() > 0 ? 1 : 0; },
        [&](int phase, int, abi::gpu::Range& record) {
            record.start = sections[phase].begin;
            record.end = sections[phase].end;
            return Geometry{grid(sections[phase].size(), call_shape.block_limit), 0};
        });
}

// --- Colouring ---

StrategyBase::Planned Colouring::plan(const op::rt::loop::Call& call, int) {
    const auto& entry = m_plans.get(call, [&] {
        op::rt::plan::PlanKey key{"colouring", call};
        return colouring_plans.get(key, [&] {
            auto plan = std::make_shared<ColouringPlan>();
            plan->colouring = op::rt::plan::colour_set(call.args, call.sections);
            upload(plan->colouring.order, plan->order_d);
            const auto& offsets = plan->colouring.offsets;
            for (std::size_t colour = 0; colour + 1 < offsets.size(); ++colour)
                plan->max_colour =
                    std::max(plan->max_colour, offsets[colour + 1] - offsets[colour]);
            if (OP_diags > 3)
                op_printf("%s: %s %d colours, %d over owned elements\n", m_name.c_str(),
                          m_executor.name(), plan->colouring.colours,
                          plan->colouring.owned_colours);
            return plan;
        });
    });
    return {&entry, {}};
}

int Colouring::max_blocks(const op::rt::loop::Call&, const void *entry, int block_limit) const {
    return grid(static_cast<const ColouringPlan *>(entry)->max_colour, block_limit);
}

std::optional<float> Colouring::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Call& call, bool timed) {
    const auto& entry = *static_cast<const ColouringPlan *>(prepared.entry);
    const auto& colouring = entry.colouring;
    auto call_shape = shape(call, prepared.entry);
    // The core's colours come first, and run while the halo arrives.
    return run_phases<abi::gpu::Range>(
        prepared, call, timed, colouring.phases(), call_shape, [](int) { return 1; },
        [&](int colour, int, abi::gpu::Range& record) {
            record.start = colouring.offsets[static_cast<std::size_t>(colour)];
            record.end = colouring.offsets[static_cast<std::size_t>(colour) + 1];
            record.order = entry.order_d;
            return Geometry{grid(record.end - record.start, call_shape.block_limit), 0};
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
    return plan_chunk_atomics(call.args, call.sections, m_family.groups, spec,
                              m_unstaged_singles,
                              op::rt::plan::chunk_options().min_compression);
}

void ChunkAtomics::add_key(op::rt::plan::PlanKey& key) const {
    key.add(m_unstaged_singles).add(op::rt::plan::chunk_options().min_compression);
}

std::string ChunkAtomics::extras(const ChunkPlan& plan) const {
    char text[96];
    std::snprintf(text, sizeof(text), "%zu refs, %zu targets (%zu unstaged)", plan.references,
                  plan.distinct_targets, plan.staging.unstaged_singles);
    return text;
}

ChunkResult ChunkColouring::make_plan(const op::rt::loop::Call& call,
                                      const ChunkSpec& spec) const {
    return plan_chunk_colouring(call.args, call.sections, m_family.groups, spec, m_wavefront,
                                options().window);
}

void ChunkColouring::add_key(op::rt::plan::PlanKey& key) const {
    key.add(m_wavefront).add(options().window);
}

std::string ChunkColouring::extras(const ChunkPlan& plan) const {
    return "up to " + std::to_string(plan.max_thread_colours) + " thread colours";
}

std::vector<op::rt::loop::Axis> ChunkAtomics::axes() {
#ifdef OP2_HIP
    return chunk_axes(
        {.variable = "UNSTAGED_SINGLES", .key = "singles", .binary = true, .defaults = {1, 0}});
#else
    return chunk_axes(
        {.variable = "UNSTAGED_SINGLES", .key = "singles", .binary = true, .defaults = {0}});
#endif
}

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, groups, block size, stride and
// variant; a chunk is a block's elements.
StrategyBase::Planned ChunkBase::plan(const op::rt::loop::Call& call, int stride) {
    const auto& entry = m_plans.get(call, [&] {
        ChunkSpec spec{m_block_size, device().shared_bytes, stride};
        op::rt::plan::PlanKey key{m_family.name, call};
        key.add(m_family.groups).add(spec.block_size).add(spec.shared_bytes).add(spec.stride);
        add_key(key);
        return chunk_plans.get(key, [&] {
            auto result = make_plan(call, spec);
            report(result);
            auto entry = std::make_shared<ChunkEntry>();
            static_cast<ChunkResult&>(*entry) = std::move(result);
            if (*entry) {
                entry->device = DevicePlan(*entry->plan);
                entry->plan->release_uploaded();
                const auto& batches = entry->plan->batches;
                for (int section = 0; section < call.sections.size(); ++section)
                    for (int batch = 0; batch < batches.batches(section); ++batch) {
                        auto [begin, end] = batches.positions(section, batch);
                        entry->max_batch = std::max(entry->max_batch, end - begin);
                    }
            }
            return entry;
        });
    });
    return {&entry, entry ? std::string_view{} : entry.rejected};
}

void ChunkBase::report(const ChunkResult& result) const {
    if (OP_diags <= 3)
        return;
    if (!result) {
        op_printf("%s: %s rejected (%s, compression %.2fx)\n", m_name.c_str(),
                  m_executor.name(), std::string{result.rejected}.c_str(),
                  result.compression);
        return;
    }
    const auto& plan = *result.plan;
    op_printf("%s: %s %s, %s\n", m_name.c_str(), m_executor.name(), plan.figures().c_str(),
              extras(plan).c_str());
}

// Reduction scratch uses the capped physical grid.  A wavefront batch's
// blocks take chunks in turn, so it needs no more of them than the device
// holds at once.
int ChunkBase::max_blocks(const op::rt::loop::Call&, const void *entry, int block_limit) const {
    const auto& chunks = *static_cast<const ChunkEntry *>(entry);
    return batch_blocks(*chunks.plan, chunks.max_batch, block_limit);
}

int ChunkBase::batch_blocks(const ChunkPlan& plan, int chunks, int block_limit) const {
    int blocks = std::min(chunks, block_limit);
    if (plan.wavefront)
        blocks = std::min(blocks, device().resident_blocks(m_block_size));
    return blocks;
}

std::optional<float> ChunkBase::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Call& call, bool timed) {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = call.sections;
    auto call_shape = shape(call, prepared.entry);

    // Every section of a wavefront call takes its positions from zero.
    int epoch = 0;
    if (plan.wavefront) {
        epoch = entry.device.next_wave_epoch();
        GPU_CHECK(gpuMemsetAsync(entry.device.wave_tickets, 0,
                                 static_cast<std::size_t>(sections.size()) * sizeof(int), 0));
    }

    return run_phases<abi::gpu::Chunk>(
        prepared, call, timed, sections.phases(), call_shape,
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
            return Geometry{batch_blocks(plan, end - begin, call_shape.block_limit),
                            static_cast<int>(bytes)};
        });
}

std::string ChunkBase::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    if (entry.plan == nullptr)
        return std::string{entry.rejected};
    const auto& plan = *entry.plan;
    return StrategyBase::statistics(prepared) + ", " + plan.figures() + ", " + extras(plan);
}

// --- Families ---

std::span<const OfferedFamily> strategy_families() {
#ifdef OP2_HIP
    constexpr bool chunk_atomics = true;
#else
    constexpr bool chunk_atomics = false;
#endif
    static const std::vector<OfferedFamily> families{
        OfferedFamily::of<Simple>("plain"),
        OfferedFamily::of<ChunkAtomics>("chunk_atomics", chunk_atomics),
        OfferedFamily::of<Simple>("atomics"),
        OfferedFamily::of<ChunkColouring>("chunk_colouring", false),
        OfferedFamily::of<Colouring>("colouring"),
    };
    return families;
}

} // namespace op::rt::gpu
