#pragma once

// The OpenMP backend's loop: the object the generated code builds per
// par_loop, holding the ladder and the tuner, and running one call through
// them over the host threads.

#include <cpu/strategy.h>
#include <loop/autotune.h>
#include <loop/strategy.h>
#include <op_lib_c.h>
#include <op_profile.h>

#include <omp.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::cpu {

// Elements per chunk when neither the translator, OP_CHUNK_*_CHUNK_SIZE nor
// OP_part_size gives one.
constexpr int default_chunk_size = 256;

class Loop {
private:
    std::string m_name;
    std::string m_profile_target;
    std::string m_profile_variant;
    LoopDescription m_description;
    op::loop::Ladder m_ladder;
    std::optional<op::loop::Autotuner> m_tuner;

    // The call under way: its sections, its threads, their globals and
    // chunk scratch.
    op::loop::Sections m_sections;
    int m_threads = 1;
    Globals m_globals;
    std::vector<std::unique_ptr<char[]>> m_scratch;
    std::size_t m_scratch_bytes = 0;

    static void release_callback(void *owner) {
        static_cast<Loop *>(owner)->release();
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

            if (is_type(arg.type, "double")) {
                op_mpi_reduce_double(&arg, reinterpret_cast<double *>(arg.data));
            } else if (is_type(arg.type, "float")) {
                op_mpi_reduce_float(&arg, reinterpret_cast<float *>(arg.data));
            } else if (is_type(arg.type, "int")) {
                op_mpi_reduce_int(&arg, reinterpret_cast<int *>(arg.data));
            } else if (is_type(arg.type, "bool")) {
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

    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, LoopDescription description)
        : m_name{name}, m_profile_target{profile_target},
          m_profile_variant{profile_variant}, m_description{description},
          m_ladder{name} {
        if (op::loop::autotune_config().report)
            m_tuner.emplace(m_ladder);
        op::loop::register_release(this, &Loop::release_callback);
    }

    ~Loop() { op::loop::unregister_release(this); }

    // --- Registration, by the generated code ---

    template<typename S, typename... Args>
    void add(op::loop::Default default_setting, Args&&... args) {
        m_ladder.add(std::make_unique<S>(*this, std::forward<Args>(args)...),
                     default_setting);
    }

    // --- For the strategies ---

    const std::string& name() const { return m_name; }
    op::loop::Ladder& ladder() { return m_ladder; }

    op::loop::Sections sections(const op::loop::Configuration& configuration) const {
        if (m_description.direct)
            return op::loop::Sections::direct(configuration.set);
        return op::loop::Sections::indirect(configuration.set,
                                            configuration.global_reduction());
    }

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
        if (bytes <= m_scratch_bytes &&
            m_scratch.size() >= static_cast<std::size_t>(m_threads))
            return;
        m_scratch_bytes = std::max(bytes, m_scratch_bytes);
        m_scratch.clear();
        for (int thread = 0; thread < m_threads; ++thread)
            m_scratch.push_back(std::make_unique<char[]>(m_scratch_bytes + 64));
    }

    char *scratch(int thread) const {
        if (m_scratch.empty())
            return nullptr;
        char *base = m_scratch[static_cast<std::size_t>(thread)].get();
        auto address = reinterpret_cast<std::uintptr_t>(base);
        return base + ((64 - address % 64) % 64);
    }

    // --- Per call ---

    // Run the loop: ready once it ran, else the floor's status, with nothing
    // run and no exchange started.
    op::loop::Status invoke(op_set set, op_arg *args, int nargs) {
        op_profile_enter_kernel(m_name.c_str(), m_profile_target.c_str(),
                                m_profile_variant.c_str());
        op_profile_enter("Init");

        op::loop::Configuration configuration{set, {args, static_cast<std::size_t>(nargs)}};
        auto selection = m_ladder.select(configuration);
        if (selection.prepared.status != op::loop::Status::ready) {
            op_profile_exit();
            op_profile_exit();
            return selection.prepared.status;
        }
        auto decision = m_tuner ? m_tuner->choose(configuration, selection)
                                : op::loop::Autotuner::Decision{
                                      selection.strategy, selection.prepared,
                                      false, nullptr};

        op_profile_next("MPI Exchanges");
        int n_exec = op_mpi_halo_exchanges(set, nargs, args, 1);
        if (n_exec == 0) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(nargs, args);
            reduce_mpi_globals(args, nargs);
            op_mpi_set_dirtybit(nargs, args);
            op_profile_exit();
            op_profile_exit();
            return op::loop::Status::ready;
        }

        op_profile_next("Prepare GBLs");
        m_sections = sections(configuration);
        m_threads = omp_get_max_threads();
        m_globals.prepare(args, nargs, m_threads);

        op_profile_next("Computation");
        op::loop::Launch launch{args, nargs, decision.timed};
        auto microseconds = decision.strategy->run(decision.prepared, launch);
        if (microseconds && m_tuner)
            m_tuner->sample(decision, *microseconds);

        op_profile_next("MPI Reduce");
        reduce_mpi_globals(args, nargs);
        op_mpi_set_dirtybit(nargs, args);
        op_profile_exit();
        op_profile_exit();
        return op::loop::Status::ready;
    }

    // At op_exit: report the configurations still being explored.
    void release() {
        if (m_tuner)
            m_tuner->flush();
    }
};

// --- The strategies' definitions that need the loop ---------------------------------

inline void StrategyBase::split(int begin, int end, const int *order,
                                op_arg *args) {
    if (end <= begin)
        return;
    // Shares are whole multiples of 16 elements, so a packed body's packs
    // stay aligned.
    const long long units = (static_cast<long long>(end) - begin + 15) / 16;
    const Globals& globals = m_loop.globals();
#pragma omp parallel num_threads(m_loop.threads())
    {
        const long long thread = omp_get_thread_num();
        const long long count = omp_get_num_threads();
        Work work;
        work.begin = static_cast<int>(
            std::min<long long>(end, begin + 16 * (units * thread / count)));
        work.end = static_cast<int>(
            std::min<long long>(end, begin + 16 * (units * (thread + 1) / count)));
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

// The phase that waits for the halo and the one after which the globals
// are combined.
inline std::pair<int, int> section_points(const op::loop::Sections& sections) {
    int wait = 0;
    int globals = 0;
    for (int i = 0; i <= sections.size(); ++i) {
        if (sections.wait_before(i))
            wait = i;
        if (i < sections.size() && sections.process_globals_after(i))
            globals = i;
    }
    return {wait, globals};
}

// --- Plain and atomics ---

inline std::optional<float> Simple::run(const op::loop::Prepared& prepared,
                                        const op::loop::Launch& launch) {
    assert(prepared.strategy == this);
    const auto& sections = m_loop.call_sections();
    auto [wait, globals] = section_points(sections);
    return run_phases(launch, sections.size(), wait, globals, [&](int section) {
        split(sections[section].begin, sections[section].end, nullptr, launch.args);
    });
}

// --- Colouring ---

inline op::loop::Prepared Colouring::prepare(
    const op::loop::Configuration& configuration, bool) {
    auto key = configuration.key();
    auto found = m_entries.find(key);
    if (found == m_entries.end()) {
        auto colouring = op::colouring::colour_set(
            configuration.set, configuration.args,
            m_loop.sections(configuration).size() == 3);
        if (OP_diags > 3)
            std::printf("%s: %s %d colours, %d over owned elements\n",
                        m_name.c_str(), m_loop.name().c_str(), colouring.colours,
                        colouring.owned_colours);
        found = m_entries.emplace(key, std::move(colouring)).first;
    }
    return prepared(op::loop::Status::ready, &found->second);
}

inline std::optional<float> Colouring::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& colouring =
        *static_cast<const op::colouring::SetColouring *>(prepared.entry);
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
        *static_cast<const op::colouring::SetColouring *>(prepared.entry);
    return std::to_string(colouring.colours) + " colours";
}

// --- The chunked strategies ---

inline ChunkBase::Entry& ChunkBase::entry_for(
    const op::loop::Configuration& configuration) {
    auto key = configuration.key();
    auto found = m_entries.find(key);
    if (found != m_entries.end())
        return found->second;

    // A thread runs a chunk on its own: no lanes, and scratch bounded only
    // so a chunk's slots stay in the caches.
    const auto& options = op::chunk::options();
    int option = m_kind == op::chunk::Kind::staged ? options.atomics_chunk_size
                                                   : options.colouring_chunk_size;
    op::chunk::Constraints constraints;
    constraints.lanes = 1;
    constraints.scratch_bytes = std::size_t{1} << 20;
    constraints.elements = m_chunk_size > 0                    ? m_chunk_size
                           : option > 0 || OP_part_size > 0 ? 0
                                                              : default_chunk_size;

    auto result = op::chunk::build_partition(m_kind, configuration,
                                             m_loop.sections(configuration),
                                             m_groups, constraints, options);
    Entry entry;
    entry.rejected = result.rejected;
    entry.partition = std::move(result.partition);
    if (entry.rejected.empty() && m_kind == op::chunk::Kind::staged)
        for (auto bytes : entry.partition->staging.chunk_scratch_bytes)
            entry.scratch_bytes = std::max(entry.scratch_bytes, bytes);

    if (OP_diags > 3) {
        if (entry.partition == nullptr)
            std::printf("%s: %s rejected (%s)\n", m_name.c_str(),
                        m_loop.name().c_str(), std::string{entry.rejected}.c_str());
        else
            std::printf("%s: %s chunk %d, %zu chunks, %zu batches, "
                        "compression %.2fx%s\n",
                        m_name.c_str(), m_loop.name().c_str(),
                        entry.partition->chunk_size, entry.partition->num_chunks(),
                        entry.partition->statistics.batches,
                        entry.partition->statistics.compression(),
                        entry.rejected.empty() ? "" : ", rejected");
    }
    return m_entries.emplace(key, std::move(entry)).first->second;
}

inline op::loop::Prepared ChunkBase::prepare(
    const op::loop::Configuration& configuration, bool) {
    auto& entry = entry_for(configuration);
    if (!entry.rejected.empty())
        return prepared(op::loop::Status::rejected, &entry, entry.rejected);
    return prepared(op::loop::Status::ready, &entry);
}

inline std::optional<float> ChunkBase::run(const op::loop::Prepared& prepared,
                                           const op::loop::Launch& launch) {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
    const auto& partition = *entry.partition;
    const auto& sections = m_loop.call_sections();
    m_loop.reserve_scratch(entry.scratch_bytes);

    auto [wait, globals] = section_points(sections);
    return run_phases(launch, sections.size(), wait, globals, [&](int section) {
        if (sections[section].size() == 0)
            return;
        const int batches = partition.batches(section);
#pragma omp parallel num_threads(m_loop.threads())
        {
            const int thread = omp_get_thread_num();
            Work work;
            work.globals = m_loop.globals().thread(thread);
            work.partition = &partition;
            work.scratch = m_loop.scratch(thread);
            for (int batch = 0; batch < batches; ++batch) {
                const auto range = partition.batch_range(section, batch);
                const int first = range.first;
                const int last = range.second;
#pragma omp for schedule(static)
                for (int position = first; position < last; ++position) {
                    work.chunk = partition.chunk_at(position);
                    work.begin = partition.source_offsets[static_cast<std::size_t>(work.chunk)];
                    work.end = partition.source_offsets[static_cast<std::size_t>(work.chunk) + 1];
                    m_body(work, launch.args);
                }
            }
        }
    });
}

inline std::string ChunkBase::statistics(const op::loop::Prepared& prepared) const {
    const auto& entry = *static_cast<const Entry *>(prepared.entry);
    const auto& partition = *entry.partition;
    char text[128];
    std::snprintf(text, sizeof(text), "chunk %d, %zu chunks, %zu batches, compression %.2f",
                  partition.chunk_size, partition.num_chunks(),
                  partition.statistics.batches, partition.statistics.compression());
    return text;
}

} // namespace op::cpu
