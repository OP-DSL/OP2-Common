#pragma once

// The parameters a loop's kernels are specialised on, and the statistics
// that move each between baked in and given at run time.

#include <extern/rapidhash.h>
#include <op2/rt/abi/descriptor.h>
#include <op2/rt/jit/options.h>
#include <op2/rt/loop/call.h>
#include <op_lib_core.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace op::rt::jit {

// How a specialised kernel has a parameter: compiled in, or given at run
// time.
enum class ParamMode {
    baked,
    runtime,
};

constexpr std::uint64_t hash_seed = RAPID_SEED;

// A backend's layout of a dat it keeps as SoA: the stride between its
// components, from an argument on it.
using DatStride = int (*)(const op_arg& arg);

class Param {
private:
    std::string m_name;
    // The value, read in place; a strategy's value is held in m_value.
    void *m_data;
    int m_value = 0;
    std::size_t m_elems;
    std::size_t m_elem_size;
    bool m_array;
    abi::ElementType m_type;
    abi::ParamSource m_source;
    // The mode before any statistics, whether they may move it, and the
    // call's.
    ParamMode m_initial = ParamMode::baked;
    bool m_fixed = true;
    ParamMode m_mode = ParamMode::baked;
    // The argument the call's value comes from, or the strategy value's slot.
    int m_index;
    std::uint64_t m_hash = 0;

public:
    // A parameter of elems elements of type, each elem_size bytes, read from
    // data; an array_arg's data and elements are each call's.
    Param(std::string_view name, abi::ElementType type, std::size_t elem_size, void *data,
          std::size_t elems, bool array, abi::ParamSource source, int index)
        : m_name{name}, m_data{data}, m_elems{elems}, m_elem_size{elem_size},
          m_array{array}, m_type{type}, m_source{source}, m_index{index} {}

    const std::string& name() const { return m_name; }
    abi::ElementType type() const { return m_type; }
    abi::ParamSource source() const { return m_source; }
    // A READ global argument, which the policy may give at run time or bake
    // in by its statistics; constants and the structural sources (strides,
    // dimensions, the strategy's values) are baked unless named otherwise.
    bool global() const {
        return m_source == abi::ParamSource::scalar_arg || m_source == abi::ParamSource::array_arg;
    }
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
    bool sized_out(const Options& options) const {
        return m_source == abi::ParamSource::array_arg &&
               (m_elems == 0 || bytes() > static_cast<std::size_t>(options.max_gbl_bytes));
    }

    // Settle the initial mode by the policy, for loop's kernels; runtime is
    // whether the mechanism can give the parameter at run time.
    void settle(std::string_view loop, bool runtime, const Options& options);

    // The call's value, from its arguments; a dat's stride as the backend
    // lays the dat out.
    void update(std::span<const op_arg> args, DatStride dat_stride);

    // The strategy's value, from its slots.
    void apply(std::span<const int> values) {
        if (m_source != abi::ParamSource::strategy)
            return;
        assert(m_index >= 0 && static_cast<std::size_t>(m_index) < values.size());
        m_value = values[static_cast<std::size_t>(m_index)];
    }

    // The value's hash, kept as last_hash: a scalar's own bits, so a zero
    // value hashes to 0, else its rapidhash.
    std::uint64_t hash() {
        if (m_array) {
            m_hash = rapidhash_withSeed(data(), bytes(), hash_seed);
            return m_hash;
        }
        switch (m_elem_size) {
        case 8: std::memcpy(&m_hash, data(), 8); break;
        case 4: {
            std::uint32_t bits;
            std::memcpy(&bits, data(), 4);
            m_hash = bits;
            break;
        }
        default: m_hash = *static_cast<const unsigned char *>(data()); break;
        }
        return m_hash;
    }
    std::uint64_t last_hash() const { return m_hash; }
};

// The calls the statistics look back over.
constexpr std::uint64_t statistics_window = 32;

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
        // The latest value, whether it changed in each of the last 32 calls
        // (the latest in bit 0), and how many of them it changed in.
        std::uint64_t last = 0;
        std::uint32_t changes = 0;
        int changed = 0;
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
        bool observe(const Param& param, const Options& options);
    };

    // A configuration's statistics, the calls observed, and its description.
    struct Histories {
        std::vector<History> params;
        std::uint64_t calls = 0;
        std::string description;
    };

    // Past this many configurations a new one keeps the initial modes.
    static constexpr std::size_t max_histories = 4096;

    std::string m_loop;
    std::vector<Param> m_params;
    DatStride m_dat_stride = nullptr;
    // Calls so far, and the one whose values were last observed.
    std::uint64_t m_calls = 0;
    std::uint64_t m_observed = ~std::uint64_t{0};
    // Whether any parameter's mode may move.
    bool m_moving = false;
    std::unordered_map<std::uint64_t, Histories> m_histories;
    std::uint64_t m_history_key = 0;
    Histories *m_history = nullptr;

    Histories *histories(const loop::Call& call);
    void observe(const loop::Call& call);

    std::size_t insert(Param param, bool runtime) {
        auto& added = m_params.emplace_back(std::move(param));
        added.settle(m_loop, runtime, jit::options());
        m_moving |= !added.fixed();
        return m_params.size() - 1;
    }

public:
    explicit ParamSet(std::string_view loop = {}) : m_loop{loop} {}
    ParamSet(const ParamSet&) = delete;

    // Add a parameter as the generated code describes it, its mode settled
    // by the policy.  Its index.
    std::size_t add(const abi::Param& param) {
        return insert(Param{param.name, param.type, param.elem_size, param.data, param.elems,
                            param.array, param.source, param.index},
                      param.runtime);
    }

    // Add a scalar parameter; runtime is whether the mechanism can give it at
    // run time.
    template<typename T>
    std::size_t add(std::string_view name, T *data, abi::ParamSource source,
                    int index = -1, bool runtime = false) {
        return insert(Param{name, abi::ElementTypes<T>::value, sizeof(T), data, 1, false,
                            source, index},
                      runtime);
    }

    // Add an array of elems elements, or of the call's dimension for an
    // array_arg.
    template<typename T>
    std::size_t add_array(std::string_view name, T *data, std::size_t elems,
                          abi::ParamSource source, int index = -1, bool runtime = false) {
        return insert(Param{name, abi::ElementTypes<T>::value, sizeof(T), data, elems, true,
                            source, index},
                      runtime);
    }

    // How the backend lays out the dats a dat_stride parameter reads.
    void set_dat_stride(DatStride stride) { m_dat_stride = stride; }

    std::size_t size() const { return m_params.size(); }
    Param& operator[](std::size_t index) { return m_params[index]; }
    const Param& operator[](std::size_t index) const { return m_params[index]; }

    // The call's values, at invoke.
    void update(std::span<const op_arg> args);

    // Whether any parameter's mode may move, so its statistics matter.
    bool moving() const { return m_moving; }

    // Whether call's configuration has its modes: none can move, or the
    // statistics have seen a window of its calls.  A mode may still move
    // later, when a value starts or stops changing.
    bool settled(const loop::Call& call) const;

    // A parameter that may move, as one configuration's statistics left it.
    struct Outcome {
        const Param *param;
        ParamMode mode;
        bool moved;
    };

    // A configuration the statistics observed: its description, calls and
    // the parameters that may move.
    struct ConfigurationModes {
        std::string_view description;
        std::uint64_t calls;
        std::vector<Outcome> params;
    };

    std::vector<ConfigurationModes> configurations() const;

    // Record the call's values in the configuration's statistics and take
    // the modes they give, once per call.
    void observe_call(const loop::Call& call) {
        if (m_observed == m_calls)
            return;
        m_observed = m_calls;
        observe(call);
    }

    // A strategy's values, by slot.
    void apply(std::span<const int> values) {
        for (auto& param : m_params)
            param.apply(values);
    }

    // The key a kernel specialised under a strategy's values is built under:
    // each baked parameter's value, hashed 64 at a time with a mask of the
    // run-time ones (a value's hash may be any 64 bits, so no mark could
    // stand in for it).
    std::uint64_t key(const loop::Call& call, std::span<const int> values) {
        observe_call(call);
        std::array<std::uint64_t, 65> words;
        std::size_t count = 0;
        std::uint64_t runtime = 0;
        std::uint64_t key = hash_seed;
        for (auto& param : m_params) {
            std::uint64_t value = 0;
            if (param.mode() == ParamMode::baked) {
                if (param.source() == abi::ParamSource::strategy) {
                    param.apply(values);
                    param.hash();
                }
                value = param.last_hash();
            } else {
                runtime |= std::uint64_t{1} << count;
            }
            words[count++] = value;
            if (count == 64) {
                words[count++] = runtime;
                key = rapidhash_withSeed(words.data(), count * sizeof(words[0]), key);
                count = 0;
                runtime = 0;
            }
        }
        words[count++] = runtime;
        return rapidhash_withSeed(words.data(), count * sizeof(words[0]), key);
    }
};

// The baked parameters as C++ constants, "static constexpr double name =
// value;" a line each, for a compiler that takes source.
std::string format_cxx(const ParamSet& params);

// A parameter's element type in C++: "double".
const char *cxx_type(abi::ElementType type);

} // namespace op::rt::jit
