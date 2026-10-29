#pragma once

// The OpenMP backend's loop: the object the generated code builds per
// par_loop, adding to the base loop the call's sections, threads, globals and
// chunk scratch over the host threads.

#include <op2/rt/cpu/strategy.h>
#include <op2/rt/loop/loop.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/plan/options.h>

#include <omp.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::rt::cpu {

class Loop : public op::rt::loop::Loop {
private:
    // The call under way: its sections, its threads, their globals and
    // chunk scratch.
    op::rt::loop::Sections m_sections;
    int m_threads = 1;
    Globals m_globals;
    std::vector<std::unique_ptr<char[]>> m_scratch;
    std::size_t m_scratch_capacity = 0;

public:
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, LoopDescription description)
        : op::rt::loop::Loop{name, profile_target, profile_variant, description.direct, false} {}

    // --- Registration, by the generated code ---

    // Add a family: a strategy S per point of its axes, built from the
    // loop, the point, the body and args.
    template<typename S, typename... Args>
    void add(std::string_view family, Body body, Args&&... args) {
        m_ladder.add(family, [&](const op::rt::loop::Point& point) {
            return std::make_unique<S>(*this, point, body, args...);
        });
    }

    // --- For the strategies ---

    const op::rt::loop::Sections& call_sections() const { return m_sections; }
    int threads() const { return m_threads; }
    const Globals& globals() const { return m_globals; }

    // Give every thread bytes of chunk scratch, aligned to a cache line.
    void reserve_scratch(std::size_t bytes) {
        if (bytes <= m_scratch_capacity &&
            m_scratch.size() >= static_cast<std::size_t>(m_threads))
            return;
        m_scratch_capacity = std::max(bytes, m_scratch_capacity);
        m_scratch.clear();
        for (int thread = 0; thread < m_threads; ++thread)
            m_scratch.push_back(std::make_unique<char[]>(m_scratch_capacity + 64));
    }

    char *scratch(int thread) const {
        if (m_scratch.empty())
            return nullptr;
        char *base = m_scratch[static_cast<std::size_t>(thread)].get();
        auto address = reinterpret_cast<std::uintptr_t>(base);
        return base + ((64 - address % 64) % 64);
    }

protected:
    // --- Per call ---

    // The call's sections, and each thread's copy of the globals.
    void begin_call(const op::rt::loop::Call& call, const op::rt::loop::Autotuner::Decision&,
                    op_arg *args, int nargs) override {
        m_sections = sections(call);
        m_threads = omp_get_max_threads();
        m_globals.prepare(args, nargs, m_threads);
    }

    // Combine the threads' copies into the arguments.
    void combine_globals(op_arg *args, int) override { m_globals.combine(args, m_name.c_str()); }
};

// --- The strategies' definitions that need the loop ---------------------------------

inline void StrategyBase::split(int begin, int end, const int *order,
                                op_arg *args) {
    if (end <= begin)
        return;
    const long long size = static_cast<long long>(end) - begin;
    const Globals& globals = m_loop.globals();
#pragma omp parallel num_threads(m_loop.threads())
    {
        const long long thread = omp_get_thread_num();
        const long long count = omp_get_num_threads();
        Work work;
        work.begin = static_cast<int>(begin + size * thread / count);
        work.end = static_cast<int>(begin + size * (thread + 1) / count);
        work.order = order;
        work.globals = globals.thread(static_cast<int>(thread));
        if (work.begin < work.end)
            m_body(work, args);
    }
}

template<typename Phase>
std::optional<float> StrategyBase::run_phases(const op::rt::loop::Launch& launch,
                                              int phases, int wait_before,
                                              int globals_after, Phase&& phase) {
    using clock = std::chrono::steady_clock;
    clock::duration elapsed{};
    m_loop.run_phases(launch, phases, wait_before, globals_after, [&](int index) {
        auto start = clock::now();
        phase(index);
        elapsed += clock::now() - start;
    });

    if (!launch.timed)
        return std::nullopt;
    return static_cast<float>(
        std::chrono::duration<double, std::micro>(elapsed).count());
}

// --- Plain and atomics ---

inline std::optional<float> Simple::run([[maybe_unused]] const op::rt::loop::Prepared& prepared,
                                        const op::rt::loop::Launch& launch) {
    assert(prepared.candidate->strategy == this);
    const auto& sections = m_loop.call_sections();
    return run_phases(launch, sections.size(), sections.wait_index(),
                      sections.globals_index(), [&](int section) {
                          split(sections[section].begin, sections[section].end,
                                nullptr, launch.args);
                      });
}

// --- Colouring ---

inline op::rt::loop::Prepared Colouring::prepare(
    const op::rt::loop::Call& call, op::rt::loop::Urgency,
    op::rt::loop::KernelKind) {
    const auto& colouring = m_plans.get(call, [&] {
        op::rt::plan::PlanKey key{"colouring", call};
        return colouring_plans().get(key, [&] {
            auto plan = std::make_shared<op::rt::plan::SetColouring>(
                op::rt::plan::colour_set(call.args, m_loop.sections(call)));
            if (OP_diags > 3)
                op_printf("%s: %s %d colours, %d over owned elements\n",
                            m_name.c_str(), m_loop.name().c_str(), plan->colours,
                            plan->owned_colours);
            return plan;
        });
    });
    return prepared(op::rt::loop::Status::ready, &colouring);
}

inline std::optional<float> Colouring::run(const op::rt::loop::Prepared& prepared,
                                           const op::rt::loop::Launch& launch) {
    const auto& colouring =
        *static_cast<const op::rt::plan::SetColouring *>(prepared.entry);
    // The core's colours come first, and run while the halo arrives.
    return run_phases(
        launch, colouring.colours, colouring.core_colours, colouring.owned_colours - 1,
        [&](int colour) {
            split(colouring.offsets[static_cast<std::size_t>(colour)],
                  colouring.offsets[static_cast<std::size_t>(colour) + 1],
                  colouring.order.data(), launch.args);
        });
}

inline std::string Colouring::statistics(const op::rt::loop::Prepared& prepared) const {
    const auto& colouring =
        *static_cast<const op::rt::plan::SetColouring *>(prepared.entry);
    return std::to_string(colouring.colours) + " colours";
}

// --- The chunked strategies ---

// The configuration's plan, the shared one when another strategy holds the
// same: the same planner, call shape, groups and chunk size.
inline const ChunkEntry& ChunkBase::entry_for(const op::rt::loop::Call& call) {
    return m_plans.get(call, [&] {
        double min_compression = op::rt::plan::chunk_options().min_compression;
        op::rt::plan::PlanKey key{m_coloured ? "chunk_colouring" : "chunk_atomics", call};
        key.add(m_chunk_size);
        if (!m_coloured)
            key.add(m_groups).add(min_compression);
        return chunk_plans().get(key, [&] {
            auto sections = m_loop.sections(call);
            auto result =
                m_coloured
                    ? plan_chunk_colouring(call.args, sections, m_chunk_size)
                    : plan_chunk_atomics(call.args, sections, m_groups, m_chunk_size,
                                         min_compression);
            auto entry = std::make_shared<ChunkEntry>();
            entry->rejected = result.rejected;
            entry->plan = std::move(result.plan);

            if (OP_diags > 3) {
                if (entry->plan == nullptr)
                    op_printf("%s: %s rejected (%s, compression %.2fx)\n",
                                m_name.c_str(), m_loop.name().c_str(),
                                std::string{entry->rejected}.c_str(), result.compression);
                else
                    op_printf("%s: %s chunk %d, %d chunks, %d batches, "
                                "compression %.2fx\n",
                                m_name.c_str(), m_loop.name().c_str(),
                                entry->plan->chunks.size, entry->plan->chunks.count(),
                                entry->plan->batches.count(), entry->plan->compression);
            }
            return entry;
        });
    });
}

inline op::rt::loop::Prepared ChunkBase::prepare(
    const op::rt::loop::Call& call, op::rt::loop::Urgency,
    op::rt::loop::KernelKind) {
    const auto& entry = entry_for(call);
    if (!entry.rejected.empty())
        return prepared(op::rt::loop::Status::unavailable, &entry, entry.rejected);
    return prepared(op::rt::loop::Status::ready, &entry);
}

inline std::optional<float> ChunkBase::run(const op::rt::loop::Prepared& prepared,
                                           const op::rt::loop::Launch& launch) {
    const auto& entry = *static_cast<const ChunkEntry *>(prepared.entry);
    const auto& plan = *entry.plan;
    const auto& sections = m_loop.call_sections();
    m_loop.reserve_scratch(plan.max_scratch_bytes);

    return run_phases(launch, sections.size(), sections.wait_index(),
                      sections.globals_index(), [&](int section) {
        const int batches = plan.batches.batches(section);
        if (batches == 0)
            return;
#pragma omp parallel num_threads(m_loop.threads())
        {
            const int thread = omp_get_thread_num();
            Work work;
            work.globals = m_loop.globals().thread(thread);
            work.plan = &plan;
            work.scratch = m_loop.scratch(thread);
            for (int batch = 0; batch < batches; ++batch) {
                const auto range = plan.batches.positions(section, batch);
                const int first = range.first;
                const int last = range.second;
#pragma omp for schedule(static)
                for (int position = first; position < last; ++position) {
                    work.chunk = plan.batches.chunk(position);
                    work.begin = plan.chunks.begin(work.chunk);
                    work.end = plan.chunks.end(work.chunk);
                    m_body(work, launch.args);
                }
            }
        }
    });
}

inline std::string ChunkBase::statistics(const op::rt::loop::Prepared& prepared) const {
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

} // namespace op::rt::cpu
