#include <op2/rt/cpu/strategy.h>

#include <op2/rt/plan/options.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace op::rt::cpu {

namespace {

// The live plans of each kind, which every strategy that would build the
// same one shares.
op::rt::plan::SharedPlans<op::rt::plan::SetColouring> colouring_plans;
op::rt::plan::SharedPlans<ChunkResult> chunk_plans;

} // namespace

// --- The base of every OpenMP strategy ----------------------------------------------

void StrategyBase::split(int begin, int end, const int *order, std::span<op_arg> args) {
    if (end <= begin)
        return;
    abi::cpu::Range range = m_loop.range();
    range.begin = begin;
    range.end = end;
    range.order = order;
    m_family.range(range, args.data());
}

template<typename F>
std::optional<float> StrategyBase::run_phases(const op::rt::loop::Call& call, bool timed,
                                              const op::rt::loop::Phases& phases, F&& phase) {
    using clock = std::chrono::steady_clock;
    clock::duration elapsed{};
    op::rt::loop::run_phases(call, phases, [&](int index) {
        auto start = clock::now();
        phase(index);
        elapsed += clock::now() - start;
    }, [&] { m_loop.combine_globals(call); });

    if (!timed)
        return std::nullopt;
    return static_cast<float>(std::chrono::duration<double, std::micro>(elapsed).count());
}

// --- Plain and atomics ---

op::rt::loop::Prepared Simple::prepare(const op::rt::loop::Call&, op::rt::loop::Urgency,
                                       op::rt::loop::KernelKind) {
    return op::rt::loop::Prepared::ready();
}

std::optional<float> Simple::run([[maybe_unused]] const op::rt::loop::Prepared& prepared,
                                 const op::rt::loop::Call& call, bool timed) {
    assert(prepared.candidate->strategy == this);
    const auto& sections = call.sections;
    return run_phases(call, timed, sections.phases(), [&](int section) {
                          split(sections[section].begin, sections[section].end, nullptr,
                                call.args);
                      });
}

// --- Colouring ---

op::rt::loop::Prepared Colouring::prepare(const op::rt::loop::Call& call,
                                          op::rt::loop::Urgency, op::rt::loop::KernelKind) {
    const auto& colouring = m_plans.get(call, [&] {
        op::rt::plan::PlanKey key{"colouring", call};
        return colouring_plans.get(key, [&] {
            auto plan = std::make_shared<op::rt::plan::SetColouring>(
                op::rt::plan::colour_set(call.args, call.sections));
            if (OP_diags > 3)
                op_printf("%s: %s %d colours, %d over owned elements\n", m_name.c_str(),
                          m_loop.name().c_str(), plan->colours, plan->owned_colours);
            return plan;
        });
    });
    return op::rt::loop::Prepared::ready(&colouring);
}

std::optional<float> Colouring::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Call& call, bool timed) {
    const auto& colouring = *static_cast<const op::rt::plan::SetColouring *>(prepared.entry);
    // The core's colours come first, and run while the halo arrives.
    return run_phases(call, timed, colouring.phases(), [&](int colour) {
                          split(colouring.offsets[static_cast<std::size_t>(colour)],
                                colouring.offsets[static_cast<std::size_t>(colour) + 1],
                                colouring.order.data(), call.args);
                      });
}

std::string Colouring::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& colouring = *static_cast<const op::rt::plan::SetColouring *>(prepared.entry);
    return std::to_string(colouring.colours) + " colours";
}

// --- The chunked strategies ---

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, chunk size and variant.
const ChunkResult& ChunkBase::entry_for(const op::rt::loop::Call& call) {
    return m_plans.get(call, [&] {
        op::rt::plan::PlanKey key{m_family.name, call};
        key.add(m_chunk_size);
        add_key(key);
        return chunk_plans.get(key, [&] {
            auto entry = std::make_shared<ChunkResult>(make_plan(call));
            if (OP_diags > 3) {
                if (!*entry)
                    op_printf("%s: %s rejected (%s, compression %.2fx)\n", m_name.c_str(),
                              m_loop.name().c_str(), std::string{entry->rejected}.c_str(),
                              entry->compression);
                else
                    op_printf("%s: %s chunk %d, %d chunks, %d batches, "
                              "compression %.2fx\n",
                              m_name.c_str(), m_loop.name().c_str(), entry->plan->chunks.size,
                              entry->plan->chunks.count(), entry->plan->batches.count(),
                              entry->plan->compression);
            }
            return entry;
        });
    });
}

op::rt::loop::Prepared ChunkBase::prepare(const op::rt::loop::Call& call,
                                          op::rt::loop::Urgency, op::rt::loop::KernelKind) {
    const auto& entry = entry_for(call);
    if (!entry)
        return op::rt::loop::Prepared::unavailable(entry.rejected, &entry);
    return op::rt::loop::Prepared::ready(&entry);
}

std::optional<float> ChunkBase::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Call& call, bool timed) {
    const auto& entry = *static_cast<const ChunkResult *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = call.sections;
    m_loop.reserve_scratch(plan.max_scratch_bytes);

    abi::cpu::Chunk chunks = m_loop.chunks();
    chunks.chunk_order = plan.batches.order.empty() ? nullptr : plan.batches.order.data();
    chunks.chunk_offsets = plan.chunks.offsets.data();
    chunks.staging = {plan.staging.stride, plan.stage_words.data(),
                     plan.staging.slot_offsets.data(), plan.staging.slot_targets.data()};

    return run_phases(call, timed, sections.phases(), [&](int section) {
                          chunks.batches = plan.batches.batches(section);
                          if (chunks.batches == 0)
                              return;
                          chunks.batch_offsets =
                              plan.batches.batch_offsets.data() +
                              plan.batches.section_batches[static_cast<std::size_t>(section)];
                          m_family.chunks(chunks, call.args.data());
                      });
}

ChunkResult ChunkAtomics::make_plan(const op::rt::loop::Call& call) const {
    return plan_chunk_atomics(call.args, call.sections, m_groups, m_chunk_size,
                              op::rt::plan::chunk_options().min_compression);
}

void ChunkAtomics::add_key(op::rt::plan::PlanKey& key) const {
    key.add(m_groups).add(op::rt::plan::chunk_options().min_compression);
}

ChunkResult ChunkColouring::make_plan(const op::rt::loop::Call& call) const {
    return plan_chunk_colouring(call.args, call.sections, m_chunk_size);
}

std::string ChunkBase::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const ChunkResult *>(prepared.entry);
    if (entry.plan == nullptr)
        return std::string{entry.rejected};
    const auto& plan = *entry.plan;
    char text[128];
    std::snprintf(text, sizeof(text), "chunk %d, %d chunks, %d batches, compression %.2f",
                  plan.chunks.size, plan.chunks.count(), plan.batches.count(),
                  plan.compression);
    return text;
}

// --- Families ---

namespace {

template<typename S>
std::unique_ptr<op::rt::loop::Strategy> make(Loop& loop, const abi::cpu::Family& family,
                                             const op::rt::loop::Point& point) {
    return std::make_unique<S>(loop, point, family);
}

} // namespace

std::span<const StrategyFamily> strategy_families() {
    static const std::vector<StrategyFamily> families{
        {{"plain", Simple::axes()}, &make<Simple>},
        {{"chunk_atomics", ChunkAtomics::axes(), false}, &make<ChunkAtomics>},
        {{"atomics", Simple::axes(), false}, &make<Simple>},
        {{"chunk_colouring", ChunkColouring::axes()}, &make<ChunkColouring>},
        {{"colouring", Colouring::axes()}, &make<Colouring>},
    };
    return families;
}

const StrategyFamily& strategy_family(std::string_view name, std::string_view loop) {
    return op::rt::loop::offered_family(strategy_families(), name, loop, "OpenMP");
}

} // namespace op::rt::cpu
