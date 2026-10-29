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
op::rt::plan::SharedPlans<ChunkEntry> chunk_plans;

op::rt::loop::Prepared prepared(op::rt::loop::Status status, const void *entry,
                                std::string_view reason = {}) {
    op::rt::loop::Prepared result;
    result.status = status;
    result.entry = entry;
    result.reason = reason;
    return result;
}

} // namespace

// --- The base of every OpenMP strategy ----------------------------------------------

void StrategyBase::split(int begin, int end, const int *order, op_arg *args) {
    if (end <= begin)
        return;
    Phase phase = m_loop.phase();
    phase.begin = begin;
    phase.end = end;
    phase.order = order;
    m_run(phase, args);
}

template<typename F>
std::optional<float> StrategyBase::run_phases(const op::rt::loop::Launch& launch, int phases,
                                              int wait_before, int globals_after, F&& phase) {
    using clock = std::chrono::steady_clock;
    clock::duration elapsed{};
    m_loop.run_phases(launch, phases, wait_before, globals_after, [&](int index) {
        auto start = clock::now();
        phase(index);
        elapsed += clock::now() - start;
    });

    if (!launch.timed)
        return std::nullopt;
    return static_cast<float>(std::chrono::duration<double, std::micro>(elapsed).count());
}

// --- Plain and atomics ---

op::rt::loop::Prepared Simple::prepare(const op::rt::loop::Call&, op::rt::loop::Urgency,
                                       op::rt::loop::KernelKind) {
    return prepared(op::rt::loop::Status::ready, nullptr);
}

std::optional<float> Simple::run([[maybe_unused]] const op::rt::loop::Prepared& prepared,
                                 const op::rt::loop::Launch& launch) {
    assert(prepared.candidate->strategy == this);
    const auto& sections = m_loop.call_sections();
    return run_phases(launch, sections.size(), sections.wait_index(),
                      sections.globals_index(), [&](int section) {
                          split(sections[section].begin, sections[section].end, nullptr,
                                launch.args);
                      });
}

// --- Colouring ---

op::rt::loop::Prepared Colouring::prepare(const op::rt::loop::Call& call,
                                          op::rt::loop::Urgency, op::rt::loop::KernelKind) {
    const auto& colouring = m_plans.get(call, [&] {
        op::rt::plan::PlanKey key{"colouring", call};
        return colouring_plans.get(key, [&] {
            auto plan = std::make_shared<op::rt::plan::SetColouring>(
                op::rt::plan::colour_set(call.args, m_loop.sections(call)));
            if (OP_diags > 3)
                op_printf("%s: %s %d colours, %d over owned elements\n", m_name.c_str(),
                          m_loop.name().c_str(), plan->colours, plan->owned_colours);
            return plan;
        });
    });
    return prepared(op::rt::loop::Status::ready, &colouring);
}

std::optional<float> Colouring::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Launch& launch) {
    const auto& colouring = *static_cast<const op::rt::plan::SetColouring *>(prepared.entry);
    // The core's colours come first, and run while the halo arrives.
    return run_phases(launch, colouring.colours, colouring.core_colours,
                      colouring.owned_colours - 1, [&](int colour) {
                          split(colouring.offsets[static_cast<std::size_t>(colour)],
                                colouring.offsets[static_cast<std::size_t>(colour) + 1],
                                colouring.order.data(), launch.args);
                      });
}

std::string Colouring::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& colouring = *static_cast<const op::rt::plan::SetColouring *>(prepared.entry);
    return std::to_string(colouring.colours) + " colours";
}

// --- The chunked strategies ---

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, groups and chunk size.
const ChunkEntry& ChunkBase::entry_for(const op::rt::loop::Call& call) {
    return m_plans.get(call, [&] {
        double min_compression = op::rt::plan::chunk_options().min_compression;
        op::rt::plan::PlanKey key{m_coloured ? "chunk_colouring" : "chunk_atomics", call};
        key.add(m_chunk_size);
        if (!m_coloured)
            key.add(m_groups).add(min_compression);
        return chunk_plans.get(key, [&] {
            auto sections = m_loop.sections(call);
            auto result = m_coloured
                              ? plan_chunk_colouring(call.args, sections, m_chunk_size)
                              : plan_chunk_atomics(call.args, sections, m_groups,
                                                   m_chunk_size, min_compression);
            auto entry = std::make_shared<ChunkEntry>();
            entry->rejected = result.rejected;
            entry->plan = std::move(result.plan);

            if (OP_diags > 3) {
                if (entry->plan == nullptr)
                    op_printf("%s: %s rejected (%s, compression %.2fx)\n", m_name.c_str(),
                              m_loop.name().c_str(), std::string{entry->rejected}.c_str(),
                              result.compression);
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
    if (!entry.rejected.empty())
        return prepared(op::rt::loop::Status::unavailable, &entry, entry.rejected);
    return prepared(op::rt::loop::Status::ready, &entry);
}

std::optional<float> ChunkBase::run(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Launch& launch) {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = m_loop.call_sections();
    m_loop.reserve_scratch(plan.max_scratch_bytes);

    Phase phase = m_loop.phase();
    phase.chunk_order = plan.batches.order.empty() ? nullptr : plan.batches.order.data();
    phase.chunk_offsets = plan.chunks.offsets.data();
    phase.scratch = m_loop.scratch();
    phase.staging = {plan.staging.stride, plan.staging.stage_words.data(),
                     plan.staging.slot_offsets.data(), plan.staging.slot_targets.data()};

    return run_phases(launch, sections.size(), sections.wait_index(),
                      sections.globals_index(), [&](int section) {
                          phase.batches = plan.batches.batches(section);
                          if (phase.batches == 0)
                              return;
                          phase.batch_offsets =
                              plan.batches.batch_offsets.data() +
                              plan.batches.section_batches[static_cast<std::size_t>(section)];
                          m_run(phase, launch.args);
                      });
}

std::string ChunkBase::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
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

std::unique_ptr<op::rt::loop::Strategy> make_strategy(Loop& loop, std::string_view family,
                                                      const op::rt::loop::Point& point,
                                                      Run run,
                                                      op::rt::plan::ArgGroups groups) {
    if (family == "plain" || family == "atomics")
        return std::make_unique<Simple>(loop, point, run);
    if (family == "colouring")
        return std::make_unique<Colouring>(loop, point, run);
    if (family == "chunk_atomics")
        return std::make_unique<ChunkAtomics>(loop, point, run, groups);
    if (family == "chunk_colouring")
        return std::make_unique<ChunkColouring>(loop, point, run);
    std::fprintf(stderr, "error: %s registers family %s, which the OpenMP backend lacks\n",
                 loop.name().c_str(), std::string{family}.c_str());
    std::exit(1);
}

} // namespace op::rt::cpu
