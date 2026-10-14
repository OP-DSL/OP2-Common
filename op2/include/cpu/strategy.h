#pragma once

// The OpenMP strategies: each schedules a loop's elements over host threads
// and calls a body the generated code supplies for one thread's share of a
// phase.  A body runs the translated kernel over its elements on the calling
// thread, so the element loop is visible to the compiler.

#include <loop/chunk_partition.h>
#include <loop/set_colouring.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace op::cpu {

class Loop;

// One thread's share of a phase, as a generated body reads it.
struct Work {
    // The elements, or under a set colouring the positions of order.
    int begin = 0;
    int end = 0;
    const int *order = nullptr;
    // The calling thread's copy of each argument kept per thread, by
    // argument index; null for the others.
    char *const *globals = nullptr;
    // chunk_atomics: the partition, the chunk and the thread's scratch.
    const op::chunk::Partition *partition = nullptr;
    int chunk = 0;
    char *scratch = nullptr;

    template<typename T>
    T *global(int arg) const {
        return reinterpret_cast<T *>(globals[arg]);
    }
};

// Run one thread's share of a phase.
using Body = void (*)(const Work& work, op_arg *args);

// A packed body's arrays: per thread, grown to the largest call's need and
// aligned to a cache line.
inline char *pack_storage(std::size_t bytes) {
    static thread_local std::vector<char> storage;
    if (storage.size() < bytes + 64)
        storage.resize(bytes + 64);
    auto address = reinterpret_cast<std::uintptr_t>(storage.data());
    return storage.data() + (64 - address % 64) % 64;
}

template<typename T>
constexpr std::size_t pack_bytes(std::size_t count) {
    return (count * sizeof(T) + 63) / 64 * 64;
}

// Carve the next count values of T out of a pack's storage.
template<typename T>
T *pack_array(char *&cursor, std::size_t count) {
    T *base = reinterpret_cast<T *>(cursor);
    cursor += pack_bytes<T>(count);
    return base;
}

// Facts about a loop that every strategy reads.
struct LoopDescription {
    bool direct;
};

// --- Per-thread globals ------------------------------------------------------------

// A copy per thread of each argument a loop reduces (OP_INC, OP_MIN,
// OP_MAX), uses as scratch (OP_WORK) or carries with a reduction
// (op_arg_info), combined into the arguments once the owned elements ran.
class Globals {
private:
    static constexpr std::size_t line = 64;

    int m_nargs = 0;
    int m_threads = 0;
    std::size_t m_stride = 0;
    std::vector<std::size_t> m_offsets;
    std::unique_ptr<char[]> m_storage;
    std::size_t m_capacity = 0;
    std::vector<char *> m_pointers;

    static bool per_thread(const op_arg& arg, const op_arg *args) {
        if (arg.argtype == OP_ARG_INFO)
            return args[arg.acc].opt != 0;
        return arg.opt != 0 && arg.argtype == OP_ARG_GBL &&
               (arg.acc == OP_INC || arg.acc == OP_MIN || arg.acc == OP_MAX ||
                arg.acc == OP_WORK);
    }

    template<typename T>
    void combine(op_arg *args, int index) const {
        auto& arg = args[index];
        T *result = reinterpret_cast<T *>(arg.data);
        for (int thread = 0; thread < m_threads; ++thread) {
            const T *value = reinterpret_cast<const T *>(
                m_pointers[static_cast<std::size_t>(thread * m_nargs + index)]);
            for (int d = 0; d < arg.dim; ++d) {
                if (arg.acc == OP_INC) {
                    result[d] += value[d];
                    continue;
                }
                bool better = arg.acc == OP_MIN ? value[d] < result[d]
                                                : value[d] > result[d];
                if (!better)
                    continue;
                result[d] = value[d];
                // The payloads travel with the thread whose value won.
                for (int i = 0; i < m_nargs; ++i)
                    if (args[i].argtype == OP_ARG_INFO && args[i].acc == index &&
                        m_offsets[static_cast<std::size_t>(i)] != SIZE_MAX)
                        std::memcpy(args[i].data,
                                    m_pointers[static_cast<std::size_t>(
                                        thread * m_nargs + i)],
                                    static_cast<std::size_t>(args[i].size));
            }
        }
    }

public:
    // Lay out and initialise every thread's copies: increments from zero,
    // the rest from the argument's value.
    void prepare(op_arg *args, int nargs, int threads) {
        m_nargs = nargs;
        m_threads = threads;
        m_offsets.assign(static_cast<std::size_t>(nargs), SIZE_MAX);
        m_stride = 0;
        for (int i = 0; i < nargs; ++i)
            if (per_thread(args[i], args)) {
                m_offsets[static_cast<std::size_t>(i)] = m_stride;
                m_stride += (static_cast<std::size_t>(args[i].size) + line - 1) /
                            line * line;
            }

        m_pointers.assign(static_cast<std::size_t>(threads * nargs), nullptr);
        if (m_stride == 0)
            return;

        std::size_t bytes = m_stride * static_cast<std::size_t>(threads) + line;
        if (m_capacity < bytes) {
            m_storage = std::make_unique<char[]>(bytes);
            m_capacity = bytes;
        }
        auto base = reinterpret_cast<std::uintptr_t>(m_storage.get());
        char *aligned = m_storage.get() + ((line - base % line) % line);

        for (int thread = 0; thread < threads; ++thread)
            for (int i = 0; i < nargs; ++i) {
                auto offset = m_offsets[static_cast<std::size_t>(i)];
                if (offset == SIZE_MAX)
                    continue;
                char *copy = aligned + m_stride * static_cast<std::size_t>(thread) +
                             offset;
                auto size = static_cast<std::size_t>(args[i].size);
                if (args[i].argtype == OP_ARG_GBL && args[i].acc == OP_INC)
                    std::memset(copy, 0, size);
                else
                    std::memcpy(copy, args[i].data, size);
                m_pointers[static_cast<std::size_t>(thread * nargs + i)] = copy;
            }
    }

    char *const *thread(int thread) const {
        return m_pointers.data() + static_cast<std::size_t>(thread * m_nargs);
    }

    // Combine every thread's reductions into the arguments, in thread order.
    void combine(op_arg *args, const char *loop) const {
        for (int i = 0; i < m_nargs; ++i) {
            const auto& arg = args[i];
            if (m_offsets[static_cast<std::size_t>(i)] == SIZE_MAX ||
                arg.argtype != OP_ARG_GBL || arg.acc == OP_WORK)
                continue;
            if (std::strcmp(arg.type, "double") == 0) {
                combine<double>(args, i);
            } else if (std::strcmp(arg.type, "float") == 0) {
                combine<float>(args, i);
            } else if (std::strcmp(arg.type, "int") == 0) {
                combine<int>(args, i);
            } else {
                std::fprintf(stderr, "error: unsupported reduction type '%s' (in %s)\n",
                             arg.type, loop);
                std::exit(1);
            }
        }
    }
};

// --- The base of every OpenMP strategy ----------------------------------------------

class StrategyBase : public op::loop::Strategy {
protected:
    Loop& m_loop;
    std::string m_name;
    Body m_body;

    StrategyBase(Loop& loop, std::string_view name, Body body)
        : m_loop{loop}, m_name{name}, m_body{body} {}

    op::loop::Prepared prepared(op::loop::Status status, const void *entry,
                                std::string_view reason = {}) {
        op::loop::Prepared result;
        result.strategy = this;
        result.status = status;
        result.entry = entry;
        result.reason = reason;
        return result;
    }

    // Split [begin, end) over the threads, a body call each.
    void split(int begin, int end, const int *order, op_arg *args);

    // Run the phases of a call: the halo wait before phase wait_before, the
    // global combine after phase globals_after, phase(index) running one
    // phase over the threads; the phases' own time when timed.
    template<typename Phase>
    std::optional<float> run_phases(const op::loop::Launch& launch, int phases,
                                    int wait_before, int globals_after,
                                    Phase&& phase);

public:
    std::string_view name() const override { return m_name; }
};

// --- Plain and atomics ------------------------------------------------------------

// Each section split over the threads; the body names the increments: none
// for plain, atomics for atomics.
class Simple : public StrategyBase {
public:
    Simple(Loop& loop, std::string_view name, Body body)
        : StrategyBase{loop, name, body} {}

    op::loop::Prepared prepare(const op::loop::Configuration&, bool) override {
        return prepared(op::loop::Status::ready, nullptr);
    }

    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
};

// --- Colouring -----------------------------------------------------------------------

// Each colour of a set colouring split over the threads.
class Colouring : public StrategyBase {
private:
    std::unordered_map<std::uint64_t, op::colouring::SetColouring> m_entries;

public:
    Colouring(Loop& loop, std::string_view name, Body body)
        : StrategyBase{loop, name, body} {}

    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               bool now) override;
    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
    std::string statistics(const op::loop::Prepared& prepared) const override;
};

// --- The chunked strategies -------------------------------------------------------

// A chunk partition: each batch's chunks shared out over the threads, a body
// call per chunk.  chunk_colouring's batches share no target, so its chunks
// run the plain body over their elements; chunk_atomics' chunks accumulate
// into the thread's scratch and flush it with atomics.
class ChunkBase : public StrategyBase {
protected:
    struct Entry {
        std::string_view rejected;
        std::unique_ptr<op::chunk::Partition> partition;
        std::size_t scratch_bytes = 0;
    };

    op::chunk::Kind m_kind;
    op::chunk::ArgGroups m_groups;
    int m_chunk_size;
    std::unordered_map<std::uint64_t, Entry> m_entries;

    ChunkBase(Loop& loop, std::string_view name, op::chunk::Kind kind,
              Body body, op::chunk::ArgGroups groups, int chunk_size)
        : StrategyBase{loop, name, body}, m_kind{kind}, m_groups{groups},
          m_chunk_size{chunk_size} {}

    Entry& entry_for(const op::loop::Configuration& configuration);

public:
    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               bool now) override;
    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
    std::string statistics(const op::loop::Prepared& prepared) const override;
};

class ChunkAtomics : public ChunkBase {
public:
    ChunkAtomics(Loop& loop, std::string_view name, Body body,
                 op::chunk::ArgGroups groups, int chunk_size)
        : ChunkBase{loop, name, op::chunk::Kind::staged, body, groups,
                    chunk_size} {}
};

class ChunkColouring : public ChunkBase {
public:
    ChunkColouring(Loop& loop, std::string_view name, Body body,
                   op::chunk::ArgGroups groups, int chunk_size)
        : ChunkBase{loop, name, op::chunk::Kind::coloured, body, groups,
                    chunk_size} {}
};

} // namespace op::cpu
