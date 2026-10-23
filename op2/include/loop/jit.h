#pragma once

// The backend-neutral half of the JIT: the policy its variables set, the
// parameters a loop's kernels are specialised on and the statistics that
// move them between baked in and given at run time, and each strategy's
// cache of built kernels.  A backend supplies the mechanism: its kernel
// type, the job that builds one, their launches, how a kernel has its
// run-time parameters, and the prior for when a build pays.

#include <extern/rapidhash.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <algorithm>
#include <atomic>
#include <bit>
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
    // OP_JIT_LEVEL: 0 builds no specialised kernel; 1 bakes in the
    // structural parameters and the constants, a constant that keeps
    // changing moving to run time, and gives READ globals at run time; 2
    // also bakes in a READ global that most calls agree on.
    int level = 2;
    bool force = false;        // OP_JIT_FORCE: specialise every kernel
    bool seq_compile = false;  // OP_JIT_SEQ_COMPILE: build on the calling thread
    bool debug = false;        // OP_JIT_DEBUG: print builds and moves, synchronise launches
    // OP_JIT_MIN_CALLS: times a key is asked for before its build starts, so
    // values that change every call never build.
    int min_calls = 9;
    // OP_JIT_MAX_THREADS: builds on threads at once, across every cache; the
    // mechanism's default when unset.
    std::optional<int> max_threads;
    // OP_JIT_VARYING: changes within a baked value's last 32 calls past
    // which it moves to run time for the rest of the run.
    int varying = 8;
    // OP_JIT_MAX_GBL_BYTES, OP_JIT_MAX_CONST_BYTES: the largest READ global
    // and constant baked in; larger ones stay at run time.
    int max_gbl_bytes = 4096;
    int max_const_bytes = 16384;
    // OP_JIT_RUNTIME_PARAMS, OP_JIT_CONSTANT_PARAMS: parameters always given
    // at run time, or always baked in, by name ("<name>" or "<loop>:<name>").
    std::vector<std::string> runtime_params;
    std::vector<std::string> constant_params;

    bool enabled() const { return level > 0; }

    // Whether a strategy's kernel is specialised: the JIT is on, and no
    // offline kernel can run, the run forces it, or the mechanism's prior
    // says a build pays.
    bool specialise(bool offline, bool prior) const {
        return enabled() && (!offline || force || prior);
    }
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
    scalar_arg,  // a scalar READ global argument
    array_arg,   // a READ global argument of any dimension, read in place
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
// What a key holds in place of a run-time parameter's value.
constexpr std::uint64_t runtime_mark = 0x9e3779b97f4a7c15ull;

class Param {
private:
    std::string m_name;
    // The value, read in place; a strategy's value is held in m_value.
    void *m_data;
    int m_value = 0;
    std::size_t m_elems;
    std::size_t m_elem_size;
    bool m_array;
    ParamType m_type;
    ParamSource m_source;
    ParamClass m_class;
    // The mode before any statistics, whether they may move it, and the
    // call's.
    ParamMode m_initial = ParamMode::baked;
    bool m_fixed = true;
    ParamMode m_mode = ParamMode::baked;
    // The argument the call's value comes from, or the strategy value's slot.
    int m_index;
    std::uint64_t m_hash = 0;

    static ParamClass class_of(ParamSource source) {
        switch (source) {
        case ParamSource::external:   return ParamClass::constant;
        case ParamSource::scalar_arg: return ParamClass::global;
        case ParamSource::array_arg:  return ParamClass::global;
        default:                      return ParamClass::structural;
        }
    }

public:
    // A parameter of elems elements, read from data; an array_arg's data
    // and elements are each call's.
    template<typename T>
    Param(std::string_view name, T *data, std::size_t elems, bool array,
          ParamSource source, int index)
        : m_name{name}, m_data{data}, m_elems{elems}, m_elem_size{sizeof(T)},
          m_array{array}, m_type{ParamTypes<T>::value}, m_source{source},
          m_class{class_of(source)}, m_index{index} {}

    const std::string& name() const { return m_name; }
    ParamType type() const { return m_type; }
    ParamSource source() const { return m_source; }
    ParamClass param_class() const { return m_class; }
    ParamMode initial() const { return m_initial; }
    bool fixed() const { return m_fixed; }
    ParamMode mode() const { return m_mode; }
    void set_mode(ParamMode mode) { m_mode = mode; }
    int index() const { return m_index; }
    std::size_t elems() const { return m_elems; }
    bool array() const { return m_array; }
    std::size_t bytes() const { return m_elems * m_elem_size; }
    const void *data() const { return m_data != nullptr ? m_data : &m_value; }

    // Whether an array_arg is at run time this call whatever its statistics:
    // inactive, or above OP_JIT_MAX_GBL_BYTES.  Its value is not hashed then.
    bool sized_out(const JitPolicy& policy) const {
        return m_source == ParamSource::array_arg &&
               (m_elems == 0 || bytes() > static_cast<std::size_t>(policy.max_gbl_bytes));
    }

    // Settle the initial mode by the policy, for loop's kernels; runtime is
    // whether the mechanism can give the parameter at run time.
    void settle(std::string_view loop, bool runtime, const JitPolicy& policy);

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

// A loop's parameters, in the order its generated code registers them,
// and per configuration of the loop the statistics of those whose mode may
// move: a baked value that changes more than OP_JIT_VARYING times in its
// last 32 calls moves to run time, and a run-time READ global baked in once
// at least 32 calls, four in five of them with one value, have changed no
// more than that; each parameter moves at most once each way.
class ParamSet {
private:
    // One parameter's statistics under one configuration.
    struct History {
        // The latest value, and whether it changed in each of the last 32
        // calls, the latest in bit 0.
        std::uint64_t last = 0;
        std::uint32_t changes = 0;
        // The value most calls have had, its calls and all calls since it
        // took the lead.
        std::uint64_t candidate = 0;
        std::uint64_t hits = 0;
        std::uint64_t total = 0;
        ParamMode mode = ParamMode::baked;
        bool seen = false;
        bool promoted = false;
        bool demoted = false;

        // Record the call's value; whether the mode moved.
        bool observe(const Param& param, const JitPolicy& policy);
    };

    // Past this many configurations a new one keeps the initial modes.
    static constexpr std::size_t max_histories = 4096;

    std::string m_loop;
    std::vector<Param> m_params;
    // Calls so far, and the one whose values were last observed.
    std::uint64_t m_calls = 0;
    std::uint64_t m_observed = ~std::uint64_t{0};
    // Whether any parameter's mode may move.
    bool m_moving = false;
    std::unordered_map<std::uint64_t, std::vector<History>> m_histories;
    std::uint64_t m_history_key = 0;
    std::vector<History> *m_history = nullptr;

    std::vector<History> *histories(std::uint64_t configuration);
    void observe(const Configuration& configuration);

    std::size_t insert(Param param, bool runtime) {
        auto& added = m_params.emplace_back(std::move(param));
        added.settle(m_loop, runtime, jit_policy());
        m_moving |= !added.fixed();
        return m_params.size() - 1;
    }

public:
    explicit ParamSet(std::string_view loop = {}) : m_loop{loop} {}
    ParamSet(const ParamSet&) = delete;

    // Add a scalar parameter, its mode settled by the policy; runtime is
    // whether the mechanism can give it at run time.  Its index.
    template<typename T>
    std::size_t add(std::string_view name, T *data, ParamSource source,
                    int index = -1, bool runtime = false) {
        return insert(Param{name, data, 1, false, source, index}, runtime);
    }

    // Add an array of elems elements, or of the call's dimension for an
    // array_arg.
    template<typename T>
    std::size_t add_array(std::string_view name, T *data, std::size_t elems,
                          ParamSource source, int index = -1, bool runtime = false) {
        return insert(Param{name, data, elems, true, source, index}, runtime);
    }

    std::size_t size() const { return m_params.size(); }
    Param& operator[](std::size_t index) { return m_params[index]; }
    const Param& operator[](std::size_t index) const { return m_params[index]; }

    // The call's values, at invoke.
    void update(op_arg *args, int nargs);

    // Whether any parameter's mode may move, so its statistics matter.
    bool moving() const { return m_moving; }

    // Record the call's values in the configuration's statistics and take
    // the modes they give, once per call.
    void observe_call(const Configuration& configuration) {
        if (m_observed == m_calls)
            return;
        m_observed = m_calls;
        observe(configuration);
    }

    // A strategy's values, by slot.
    void apply(std::span<const int> values) {
        for (auto& param : m_params)
            param.apply(values);
    }

    // The key a kernel specialised under a strategy's values is built under:
    // each baked parameter's value, each run-time one marked as such.
    std::uint64_t key(const Configuration& configuration, std::span<const int> values) {
        observe_call(configuration);
        std::uint64_t key = hash_seed;
        for (auto& param : m_params) {
            std::uint64_t value = runtime_mark;
            if (param.mode() == ParamMode::baked) {
                if (param.source() == ParamSource::strategy) {
                    param.apply(values);
                    param.hash();
                }
                value = param.last_hash();
            }
            key = rapidhash_withSeed(&value, sizeof(value), key);
        }
        return key;
    }
};

// The baked parameters as C++ constants, "static constexpr double name =
// value;" a line each, for a compiler that takes source.
std::string format_cxx(const ParamSet& params);

// A parameter's element type in C++: "double".
const char *cxx_type(ParamType type);

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
