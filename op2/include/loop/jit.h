#pragma once

// The backend-neutral half of the JIT: the policy its variables set, the
// parameters a loop's kernels are specialised on, and each strategy's cache
// of built kernels.  A backend supplies the mechanism: its kernel type, the
// job that builds one, their launches, and the prior for when a build pays.

#include <extern/rapidhash.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace op::loop {

// --- Policy -------------------------------------------------------------------

struct JitPolicy {
    bool enable = true;        // OP_JIT_ENABLE
    bool force = false;        // OP_JIT_FORCE: specialise every kernel
    bool seq_compile = false;  // OP_JIT_SEQ_COMPILE: build on the calling thread
    bool debug = false;        // OP_JIT_DEBUG: print builds, synchronise launches
    // OP_JIT_MIN_CALLS: times a key is asked for before its build starts, so
    // values that change every call never build.
    int min_calls = 9;
    // OP_JIT_MAX_THREADS: builds on threads at once, across every cache; the
    // mechanism's default when unset.
    std::optional<int> max_threads;
};

const JitPolicy& jit_policy();
// Read by init.
void init_jit_policy();

// Take one of limit build slots, shared by every cache; whether one was free.
bool take_build_slot(int limit);
void release_build_slot();

// --- Parameters ---------------------------------------------------------------

enum class ParamType {
    i32,
    i64,
    f32,
    f64,
    logical,
};

template<typename T> struct ParamTypes {};
template<> struct ParamTypes<int>     { static const ParamType value = ParamType::i32; };
template<> struct ParamTypes<int64_t> { static const ParamType value = ParamType::i64; };
template<> struct ParamTypes<float>   { static const ParamType value = ParamType::f32; };
template<> struct ParamTypes<double>  { static const ParamType value = ParamType::f64; };
template<> struct ParamTypes<bool>    { static const ParamType value = ParamType::logical; };

// Where a parameter's value comes from: the application, the call, or the
// strategy.
enum class ParamSource {
    external,    // a constant the application sets
    scalar_arg,  // a scalar global argument
    dat_stride,  // a dat's SoA stride
    dat_dim,     // a dat's run-time dimension
    strategy,    // a value the strategy fixes, in its backend's slot
};

// What a parameter is to the policy.
enum class ParamClass {
    structural,  // strides, dimensions and the strategy's values: always baked
    constant,    // an application constant
    global,      // a READ global argument
};

// How a specialised kernel has a parameter: compiled in, or given at run
// time.
enum class ParamMode {
    baked,
    runtime,
};

constexpr std::uint64_t hash_seed = RAPID_SEED;

class Param {
private:
    std::string m_name;
    // The value, read in place; a strategy's value is held in m_value.
    void *m_data;
    int m_value = 0;
    std::size_t m_elems;
    std::size_t m_elem_size;
    ParamType m_type;
    ParamSource m_source;
    ParamClass m_class;
    ParamMode m_mode = ParamMode::baked;
    // The argument the call's value comes from, or the strategy value's slot.
    int m_index;
    std::uint64_t m_hash = 0;

    static ParamClass class_of(ParamSource source) {
        switch (source) {
        case ParamSource::external:   return ParamClass::constant;
        case ParamSource::scalar_arg: return ParamClass::global;
        default:                      return ParamClass::structural;
        }
    }

public:
    // A parameter of elems elements, an array when more than one, read from
    // data.
    template<typename T>
    Param(std::string_view name, T *data, std::size_t elems, ParamSource source,
          int index)
        : m_name{name}, m_data{data}, m_elems{elems}, m_elem_size{sizeof(T)},
          m_type{ParamTypes<T>::value}, m_source{source},
          m_class{class_of(source)}, m_index{index} {}

    const std::string& name() const { return m_name; }
    ParamType type() const { return m_type; }
    ParamSource source() const { return m_source; }
    ParamClass param_class() const { return m_class; }
    ParamMode mode() const { return m_mode; }
    std::size_t elems() const { return m_elems; }
    bool array() const { return m_elems > 1; }
    std::size_t bytes() const { return m_elems * m_elem_size; }
    const void *data() const { return m_data != nullptr ? m_data : &m_value; }

    // The call's value, from its arguments.
    void update(op_arg *args, int nargs);

    // The strategy's value, from its slots.
    void apply(std::span<const int> values) {
        if (m_source != ParamSource::strategy)
            return;
        assert(m_index >= 0 && static_cast<std::size_t>(m_index) < values.size());
        m_value = values[static_cast<std::size_t>(m_index)];
    }

    // The value's hash, kept as last_hash.
    std::uint64_t hash() {
        m_hash = rapidhash_withSeed(data(), bytes(), hash_seed);
        return m_hash;
    }
    std::uint64_t last_hash() const { return m_hash; }
};

// A loop's parameters, in the order its generated code registers them.
class ParamSet {
private:
    std::vector<Param> m_params;

public:
    // Add a parameter; its index.
    template<typename T>
    std::size_t add(std::string_view name, T *data, std::size_t elems,
                    ParamSource source, int index = -1) {
        m_params.emplace_back(name, data, elems, source, index);
        return m_params.size() - 1;
    }

    std::size_t size() const { return m_params.size(); }
    Param& operator[](std::size_t index) { return m_params[index]; }
    const Param& operator[](std::size_t index) const { return m_params[index]; }

    // The call's values, at invoke.
    void update(op_arg *args, int nargs);

    // A strategy's values, by slot.
    void apply(std::span<const int> values) {
        for (auto& param : m_params)
            param.apply(values);
    }

    // The key a kernel specialised on the current values is built under:
    // the baked parameters' hashes, combined.
    std::uint64_t hash() {
        std::uint64_t hash = hash_seed;
        for (auto& param : m_params) {
            if (param.mode() != ParamMode::baked)
                continue;
            std::uint64_t value = param.hash();
            hash = rapidhash_withSeed(&value, sizeof(value), hash);
        }
        return hash;
    }
};

// The baked parameters as C++ constants, "static constexpr double name =
// value;" a line each, for a compiler that takes source.
std::string format_cxx(const ParamSet& params);

// --- Builds -------------------------------------------------------------------

// Where a key's build stands.
enum class BuildState {
    unseen,  // never asked for, or evicted before its build started
    seen,    // asked for, its build not started
    building,
    ready,
    failed,
};

// A strategy's built kernels, one per key (the hash of the values each is
// specialised on), and when each is built.
template<typename Kernel>
class JitCache {
public:
    struct Acquired {
        Status status = Status::unavailable;
        // The built kernel; none to run the offline one.
        Kernel *kernel = nullptr;
        // Why it is not ready: "compiling" or "build_failed".
        std::string_view reason;
    };

private:
    struct Entry {
        // Times asked for while its build had not started.
        std::size_t calls = 0;
        // When it was last asked for, on m_clock.
        std::uint64_t used = 0;
        std::atomic<BuildState> state{BuildState::seen};
        std::thread thread;
        std::unique_ptr<Kernel> kernel;
    };

    // Past this many entries the least recently asked-for half of those not
    // started go, so values that change every call stay bounded.
    static constexpr std::size_t max_entries = 256;

    std::string m_name;
    int m_threads;
    std::unordered_map<std::uint64_t, Entry> m_entries;
    std::uint64_t m_clock = 0;

    // The entry for key, marked as asked for now.
    Entry& entry_for(std::uint64_t key) {
        auto [found, inserted] = m_entries.try_emplace(key);
        found->second.used = ++m_clock;
        if (inserted && m_entries.size() > max_entries)
            evict();
        return found->second;
    }

    // Drop the least recently asked-for half of the entries not started; the
    // newest is never among them.
    void evict() {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> idle;
        for (const auto& [key, entry] : m_entries)
            if (entry.state.load(std::memory_order_relaxed) == BuildState::seen)
                idle.emplace_back(entry.used, key);
        auto middle = idle.begin() + static_cast<std::ptrdiff_t>(idle.size() / 2);
        std::nth_element(idle.begin(), middle, idle.end());
        for (auto item = idle.begin(); item != middle; ++item)
            m_entries.erase(item->second);
    }

    static void finish(Entry& entry, std::unique_ptr<Kernel> kernel) {
        entry.kernel = std::move(kernel);
        entry.state.store(entry.kernel != nullptr ? BuildState::ready
                                                  : BuildState::failed,
                          std::memory_order_release);
    }

public:
    // name, the kernel's, for messages; threads, the mechanism's default for
    // OP_JIT_MAX_THREADS.
    JitCache(std::string_view name, int threads) : m_name{name}, m_threads{threads} {}
    JitCache(const JitCache&) = delete;
    ~JitCache() { join(); }

    // The kernel for key once built.  Its build starts once key is asked for
    // OP_JIT_MIN_CALLS times, at once for Urgency::now and wait, while a
    // build slot is free; it runs here when urgency is wait and no offline
    // kernel can run meanwhile, or under OP_JIT_SEQ_COMPILE, else on a
    // thread.  Without the kernel: ready with none when offline (the offline
    // kernel is usable), else pending until it is built, unavailable once
    // its build failed.  make_job() is called here when a build starts, and
    // returns the job the build runs: it returns the kernel, or none when
    // the build failed.
    template<typename MakeJob>
    Acquired acquire(std::uint64_t key, Urgency urgency, bool offline,
                     MakeJob&& make_job) {
        auto found = m_entries.find(key);
        if (found != m_entries.end() &&
            found->second.state.load(std::memory_order_acquire) == BuildState::ready)
            return {Status::ready, found->second.kernel.get(), {}};

        auto& entry = entry_for(key);
        ++entry.calls;
        const auto& policy = jit_policy();
        // Waiting is for a strategy with nothing else to run.
        bool here = (urgency == Urgency::wait && !offline) || policy.seq_compile;
        bool due = urgency != Urgency::later ||
                   entry.calls >= static_cast<std::size_t>(policy.min_calls);
        if (due && entry.state.load(std::memory_order_relaxed) == BuildState::seen &&
            (here || take_build_slot(policy.max_threads.value_or(m_threads)))) {
            if (policy.debug)
                std::printf("compiling %s for hash %lx\n", m_name.c_str(),
                            static_cast<unsigned long>(key));
            auto job = make_job();
            if (here) {
                finish(entry, job());
            } else {
                entry.state.store(BuildState::building, std::memory_order_relaxed);
                entry.thread = std::thread([&entry, job = std::move(job)]() mutable {
                    finish(entry, job());
                    release_build_slot();
                });
            }
        }
        if (here && entry.thread.joinable())
            entry.thread.join();

        switch (entry.state.load(std::memory_order_acquire)) {
        case BuildState::ready:
            return {Status::ready, entry.kernel.get(), {}};
        case BuildState::failed:
            return offline ? Acquired{Status::ready, nullptr, {}}
                           : Acquired{Status::unavailable, nullptr, "build_failed"};
        default:
            return offline ? Acquired{Status::ready, nullptr, {}}
                           : Acquired{Status::pending, nullptr, "compiling"};
        }
    }

    // Where key's build stands; no side effects.
    BuildState state(std::uint64_t key) const {
        auto found = m_entries.find(key);
        return found == m_entries.end()
                   ? BuildState::unseen
                   : found->second.state.load(std::memory_order_acquire);
    }

    // Wait for every build still running.
    void join() {
        for (auto& [key, entry] : m_entries)
            if (entry.thread.joinable())
                entry.thread.join();
    }
};

} // namespace op::loop
