#pragma once

// The OpenMP backend's loop: the object the generated code builds per
// par_loop, adding to the base loop the call's sections, threads, globals and
// chunk scratch, and running one call through them over the host threads.

#include <cpu/strategy.h>
#include <loop/loop.h>
#include <loop/strategy.h>
#include <op_lib_c.h>
#include <op_profile.h>
#include <plan/options.h>

#include <omp.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::cpu {

class Loop : public op::loop::Loop {
private:
    // The call under way: its sections, its threads, their globals and
    // chunk scratch.
    op::loop::Sections m_sections;
    int m_threads = 1;
    Globals m_globals;
    std::vector<std::unique_ptr<char[]>> m_scratch;
    std::size_t m_scratch_capacity = 0;

public:
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, LoopDescription description)
        : op::loop::Loop{name, profile_target, profile_variant, description.direct} {}

    // --- Registration, by the generated code ---

    // Add a family: a strategy S per point of its axes, built from the
    // loop, the point, the body and args.
    template<typename S, typename... Args>
    void add(std::string_view family, Body body, Args&&... args) {
        m_ladder.add(family, [&](const op::loop::Point& point) {
            return std::make_unique<S>(*this, point, body, args...);
        });
    }

    // --- For the strategies ---

    const op::loop::Sections& call_sections() const { return m_sections; }
    int threads() const { return m_threads; }
    const Globals& globals() const { return m_globals; }

    void combine_globals(op_arg *args) {
        op_profile_next("Process GBLs");
        m_globals.combine(args, m_name.c_str());
        op_profile_next("Computation");
    }

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

    // --- Per call ---

    // Run the loop.
    void invoke(op_set set, op_arg *args, int nargs) {
        op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                                m_profile_variant.c_str());
        // The exchange decides what the call runs, so it comes first, and
        // the planning overlaps it.
        op_profile_enter("MPI Exchanges");
        int n_exec = op_mpi_halo_exchanges(set, nargs, args, 1);
        if (n_exec == 0) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(nargs, args);
            op::loop::reduce_mpi_globals(args, nargs, m_name.c_str());
            op_mpi_set_dirtybit(nargs, args);
            op_profile_exit();
            op_profile_exit();
            return;
        }

        op_profile_next("Init");
        op::loop::Configuration configuration{
            set, {args, static_cast<std::size_t>(nargs)}, n_exec};
        auto decision = decide(configuration, args, nargs);

        op_profile_next("Prepare GBLs");
        m_sections = sections(configuration);
        m_threads = omp_get_max_threads();
        m_globals.prepare(args, nargs, m_threads);

        op_profile_next("Computation");
        run(decision, args, nargs);

        op_profile_next("MPI Reduce");
        op::loop::reduce_mpi_globals(args, nargs, m_name.c_str());
        op_mpi_set_dirtybit(nargs, args);
        op_profile_exit();
        op_profile_exit();
    }
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
std::optional<float> StrategyBase::run_phases(const op::loop::Launch& launch,
                                              int phases, int wait_before,
                                              int globals_after, Phase&& phase) {
    using clock = std::chrono::steady_clock;
    clock::duration elapsed{};
    for (int index = 0; index < phases; ++index) {
        if (index == wait_before) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(launch.nargs, launch.args);
            op_profile_next("Computation");
        }

        auto start = clock::now();
        phase(index);
        elapsed += clock::now() - start;

        if (index == globals_after)
            m_loop.combine_globals(launch.args);
    }

    if (phases == wait_before) {
        op_profile_next("MPI Wait");
        op_mpi_wait_all(launch.nargs, launch.args);
        op_profile_next("Computation");
    }

    if (!launch.timed)
        return std::nullopt;
    return static_cast<float>(
        std::chrono::duration<double, std::micro>(elapsed).count());
}

// --- Plain and atomics ---

inline std::optional<float> Simple::run(const op::loop::Prepared& prepared,
                                        const op::loop::Launch& launch) {
    assert(prepared.strategy == this);
    const auto& sections = m_loop.call_sections();
    return run_phases(launch, sections.size(), sections.wait_index(),
                      sections.globals_index(), [&](int section) {
                          split(sections[section].begin, sections[section].end,
                                nullptr, launch.args);
                      });
}

// --- Colouring ---

inline op::loop::Prepared Colouring::prepare(
    const op::loop::Configuration& configuration, op::loop::Urgency,
    op::loop::KernelKind) {
    auto key = configuration.key();
    auto found = m_entries.find(key);
    if (found == m_entries.end()) {
        auto colouring = op::plan::colour_set(configuration.args,
                                              m_loop.sections(configuration));
        if (OP_diags > 3)
            op_printf("%s: %s %d colours, %d over owned elements\n",
                        m_name.c_str(), m_loop.name().c_str(), colouring.colours,
                        colouring.owned_colours);
        found = m_entries.emplace(key, std::move(colouring)).first;
    }
    return prepared(op::loop::Status::ready, &found->second);
}

inline std::optional<float> Colouring::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& colouring =
        *static_cast<const op::plan::SetColouring *>(prepared.entry);
    // Every colour can hold halo-dependent elements, so the wait comes first.
    return run_phases(
        launch, colouring.colours, 0, colouring.owned_colours - 1, [&](int colour) {
            split(colouring.offsets[static_cast<std::size_t>(colour)],
                  colouring.offsets[static_cast<std::size_t>(colour) + 1],
                  colouring.order.data(), launch.args);
        });
}

inline std::string Colouring::statistics(const op::loop::Prepared& prepared) const {
    const auto& colouring =
        *static_cast<const op::plan::SetColouring *>(prepared.entry);
    return std::to_string(colouring.colours) + " colours";
}

// --- The chunked strategies ---

inline ChunkBase::Entry& ChunkBase::entry_for(
    const op::loop::Configuration& configuration) {
    auto key = configuration.key();
    auto found = m_entries.find(key);
    if (found != m_entries.end())
        return found->second;

    auto sections = m_loop.sections(configuration);
    auto result =
        m_coloured
            ? plan_chunk_colouring(configuration.args, sections, m_chunk_size)
            : plan_chunk_atomics(configuration.args, sections, m_groups, m_chunk_size,
                                 op::plan::chunk_options().min_compression);
    Entry entry;
    entry.rejected = result.rejected;
    entry.plan = std::move(result.plan);

    if (OP_diags > 3) {
        if (entry.plan == nullptr)
            op_printf("%s: %s rejected (%s, compression %.2fx)\n", m_name.c_str(),
                        m_loop.name().c_str(), std::string{entry.rejected}.c_str(),
                        result.compression);
        else
            op_printf("%s: %s chunk %d, %d chunks, %d batches, "
                        "compression %.2fx\n",
                        m_name.c_str(), m_loop.name().c_str(),
                        entry.plan->chunks.size, entry.plan->chunks.count(),
                        entry.plan->batches.count(), entry.plan->compression);
    }
    return m_entries.emplace(key, std::move(entry)).first->second;
}

inline op::loop::Prepared ChunkBase::prepare(
    const op::loop::Configuration& configuration, op::loop::Urgency,
    op::loop::KernelKind) {
    auto& entry = entry_for(configuration);
    if (!entry.rejected.empty())
        return prepared(op::loop::Status::rejected, &entry, entry.rejected);
    return prepared(op::loop::Status::ready, &entry);
}

inline std::optional<float> ChunkBase::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
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

inline std::string ChunkBase::statistics(const op::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
    if (entry.plan == nullptr)
        return std::string{entry.rejected};
    const auto& plan = *entry.plan;
    char text[128];
    std::snprintf(text, sizeof(text), "chunk %d, %d chunks, %d batches, compression %.2f",
                  plan.chunks.size, plan.chunks.count(), plan.batches.count(),
                  plan.compression);
    return text;
}

} // namespace op::cpu
