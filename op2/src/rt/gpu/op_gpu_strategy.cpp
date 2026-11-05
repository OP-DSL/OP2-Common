#include <op2/rt/gpu/strategy.h>

#include <op2/rt/gpu/heuristics.h>
#include <op2/rt/gpu/upload.h>

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/context.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace op::rt::gpu {

op::rt::loop::Axis block_size_axis() {
    return {.variable = "BLOCK_SIZES", .key = "block_size", .defaults = defaults().block_sizes};
}

ColouringPlan::~ColouringPlan() { release(order_d); }

ChunkEntry::ChunkEntry(ChunkResult result) : ChunkResult{std::move(result)} {
    if (plan == nullptr)
        return;
    device = DevicePlan(*plan);
    plan->release_uploaded();
}

// --- The base of every GPU strategy ------------------------------------------------

StrategyBase::StrategyBase(Executor& executor, const op::rt::loop::Point& point,
                           const abi::gpu::Family& family)
    : m_executor{executor}, m_name{point.name}, m_family{family},
      m_block_size{point.value("block_size")},
      m_values(static_cast<std::size_t>(executor.params().slots()), 0),
      m_offline{family.offline, executor.calls()},
      m_cache{family.kernel, executor.context().jit, executor.context().builds} {
    for (const auto& param : executor.params().params()) {
        auto slot = static_cast<std::size_t>(param.slot());
        if (param.source() == abi::ParamSource::strategy && point.has(param.key()))
            m_values[slot] = point.value(param.key());
        else if (param.source() == abi::ParamSource::global_stride)
            m_global_stride_slot = param.slot();
    }
}

op::rt::loop::Estimate StrategyBase::estimate(op::rt::loop::KernelKind kind) const {
    KernelFacts facts;
    facts.offline = offline_usable();
    facts.resources = m_offline.resources();
    facts.jit = m_family.source != nullptr && m_family.source[0] != '\0';
    return gpu::estimate(m_executor.backend().heuristics, facts, kind);
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
    auto modes = params.modes();
    if (m_global_stride_slot >= 0)
        m_values[static_cast<std::size_t>(m_global_stride_slot)] =
            shape(call, planned.entry).global_stride;
    auto key = params.key(modes, m_values);
    auto acquired = m_cache.acquire(key, urgency, [&] {
        return m_executor.compiler().job(params, modes, m_values, m_family);
    });
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
    if (prepared.kernel != nullptr)
        static_cast<JitKernel *>(prepared.kernel)
            ->launch(geometry.blocks, m_block_size, arguments, geometry.shared_bytes);
    else
        m_offline.launch(geometry.blocks, m_block_size, arguments, geometry.shared_bytes);
    // OP_JIT_DEBUG: each launch completes before the next, so a fault names its kernel.
    if (m_executor.context().jit.debug)
        GPU_CHECK(gpuStreamSynchronize(0));
}

template<typename Record, typename Launches, typename Fill>
std::optional<float> StrategyBase::run_phases(const op::rt::loop::Prepared& prepared,
                                              const op::rt::loop::Call& call, bool timed,
                                              const op::rt::loop::Phases& phases,
                                              const Shape& shape, Launches&& launches,
                                              Fill&& fill) {
    m_executor.prepare_globals(prepared, call, shape.global_stride);
    // The kernel reads the call's run-time values from device symbols.
    if (prepared.kernel != nullptr)
        static_cast<JitKernel *>(prepared.kernel)->upload(m_executor.params());
    else
        m_executor.upload_symbols();
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
            record.begin = sections[phase].begin;
            record.end = sections[phase].end;
            return Geometry{grid(sections[phase].size(), call_shape.block_limit), 0};
        });
}

// --- Colouring ---

StrategyBase::Planned Colouring::plan(const op::rt::loop::Call& call, int) {
    const auto& entry = m_plans.get(call, [&] {
        op::rt::plan::PlanKey key{"colouring", call};
        return m_executor.backend().colouring_plans.get(key, [&] {
            auto plan = std::make_shared<ColouringPlan>();
            plan->colouring = op::rt::plan::colour_set(call.args, call.sections);
            upload(plan->colouring.order, plan->order_d);
            plan->colouring.order = {};
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
            record.begin = colouring.offsets[static_cast<std::size_t>(colour)];
            record.end = colouring.offsets[static_cast<std::size_t>(colour) + 1];
            record.order = entry.order_d;
            return Geometry{grid(record.end - record.begin, call_shape.block_limit), 0};
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
                              m_executor.backend().options.min_compression);
}

void ChunkAtomics::add_key(op::rt::plan::PlanKey& key) const {
    key.add(m_unstaged_singles).add(m_executor.backend().options.min_compression);
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
                                m_executor.backend().options.window);
}

void ChunkColouring::add_key(op::rt::plan::PlanKey& key) const {
    key.add(m_wavefront).add(m_executor.backend().options.window);
}

std::string ChunkColouring::extras(const ChunkPlan& plan) const {
    return "up to " + std::to_string(plan.max_thread_colours) + " thread colours";
}

std::vector<op::rt::loop::Axis> ChunkAtomics::axes() {
    return chunk_axes({.variable = "UNSTAGED_SINGLES",
                       .key = "singles",
                       .binary = true,
                       .defaults = defaults().unstaged_singles});
}

std::vector<op::rt::loop::Axis> ChunkColouring::axes() {
    return chunk_axes({.variable = "WAVEFRONT",
                       .key = "wavefront",
                       .binary = true,
                       .defaults = defaults().wavefront});
}

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, groups, block size, stride and
// variant; a chunk is a block's elements.
StrategyBase::Planned ChunkBase::plan(const op::rt::loop::Call& call, int stride) {
    const auto& entry = m_plans.get(call, [&] {
        ChunkSpec spec{.block_size = m_block_size,
                       .shared_bytes = m_executor.backend().device.shared_bytes,
                       .stride = stride};
        op::rt::plan::PlanKey key{m_family.name, call};
        key.add(m_family.groups).add(spec.block_size).add(spec.shared_bytes).add(spec.stride);
        add_key(key);
        return m_executor.backend().chunk_plans.get(key, [&] {
            auto result = make_plan(call, spec);
            report(result);
            return std::make_shared<ChunkEntry>(std::move(result));
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
    const auto& plan = *static_cast<const ChunkEntry *>(entry)->plan;
    return batch_blocks(plan, plan.max_batch, block_limit);
}

int ChunkBase::batch_blocks(const ChunkPlan& plan, int chunks, int block_limit) const {
    int blocks = std::min(chunks, block_limit);
    if (plan.wavefront)
        blocks = std::min(blocks, m_executor.backend().device.resident_blocks(m_block_size));
    return blocks;
}

std::optional<float> ChunkBase::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Call& call, bool timed) {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = call.sections;
    auto call_shape = shape(call, prepared.entry);

    entry.device.begin_wavefront_call();
    return run_phases<abi::gpu::Chunk>(
        prepared, call, timed, sections.phases(), call_shape,
        [&](int section) { return plan.batches.batches(section); },
        [&](int section, int batch, abi::gpu::Chunk& record) {
            auto [begin, end] = plan.batches.positions(section, batch);
            entry.device.fill(record, section);
            record.chunk_begin = begin;
            record.chunk_end = end;
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
    static const std::vector<OfferedFamily> families{
        OfferedFamily::of<Simple>("plain"),
        OfferedFamily::of<ChunkAtomics>("chunk_atomics", defaults().chunk_atomics),
        OfferedFamily::of<Simple>("atomics"),
        OfferedFamily::of<ChunkColouring>("chunk_colouring", defaults().chunk_colouring),
        OfferedFamily::of<Colouring>("colouring"),
    };
    return families;
}

} // namespace op::rt::gpu
