#pragma once

// The parameters a loop's kernels are specialised on, and the statistics
// that move each between baked in and given at run time.

#include <extern/rapidhash.h>
#include <op2/rt/abi/descriptor.h>
#include <op2/rt/hasher.h>
#include <op2/rt/jit/options.h>
#include <op2/rt/loop/call.h>
#include <op_lib_core.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
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
    // As the generated code describes it, its name and value as long-lived
    // as the descriptor.
    abi::Param m_row;
    // The value: an array_arg's data and elements are each call's.
    void *m_data;
    std::size_t m_elems;
    // The mode before any statistics, whether they may move it, and the
    // call's.
    ParamMode m_initial = ParamMode::baked;
    bool m_fixed = true;
    ParamMode m_mode = ParamMode::baked;
    std::uint64_t m_hash = 0;

    // The value's hash: a scalar's own bits, so a zero value hashes to 0,
    // else its rapidhash.
    void rehash() {
        if (array()) {
            m_hash = rapidhash_withSeed(data(), bytes(), hash_seed);
            return;
        }
        switch (elem_size()) {
        case 8: std::memcpy(&m_hash, data(), 8); break;
        case 4: {
            std::uint32_t bits;
            std::memcpy(&bits, data(), 4);
            m_hash = bits;
            break;
        }
        default: m_hash = *static_cast<const unsigned char *>(data()); break;
        }
    }

public:
    explicit Param(const abi::Param& row) : m_row{row}, m_data{row.data}, m_elems{row.elems} {}

    const char *name() const { return m_row.name; }
    abi::ElementType type() const { return m_row.type; }
    abi::ParamSource source() const { return m_row.source; }
    // A READ global argument, which the policy may give at run time or bake
    // in by its statistics; constants and the structural sources (strides,
    // dimensions, the strategy's values) are baked unless named otherwise.
    bool global() const {
        return source() == abi::ParamSource::scalar_arg ||
               source() == abi::ParamSource::array_arg;
    }
    ParamMode initial() const { return m_initial; }
    bool fixed() const { return m_fixed; }
    ParamMode mode() const { return m_mode; }
    void set_mode(ParamMode mode) { m_mode = mode; }
    // The argument the call's value comes from, or the strategy value's slot.
    int index() const { return m_row.index; }
    std::size_t elems() const { return m_elems; }
    std::size_t elem_size() const { return m_row.elem_size; }
    bool array() const { return m_row.array; }
    std::size_t bytes() const { return m_elems * elem_size(); }
    const void *data() const { return m_data; }

    // Whether an array_arg is at run time this call whatever its statistics:
    // inactive, or above OP_JIT_MAX_GBL_BYTES.  Its value is not hashed then.
    bool sized_out(const Options& options) const {
        return source() == abi::ParamSource::array_arg &&
               (m_elems == 0 || bytes() > static_cast<std::size_t>(options.max_gbl_bytes));
    }

    // Settle the initial mode by the policy, for loop's kernels.
    void settle(std::string_view loop, const Options& options);

    // The call's value, from its arguments, and its hash; a dat's stride as
    // the backend lays the dat out.  A strategy's value comes with each
    // kernel's key, and one sized out is not hashed.
    void update(std::span<const op_arg> args, DatStride dat_stride, const Options& options);

    // The call's hash; a strategy's value is hashed with the key instead.
    std::uint64_t hash() const { return m_hash; }
};

// The calls the statistics look back over.
constexpr std::uint64_t statistics_window = 32;

// A loop's parameters, in the order its generated code registers them,
// and per configuration of the loop the statistics of those whose mode may
// move: a parameter is baked in while it takes at most OP_JIT_MAX_VALUES
// values, a baked one moving to run time at its next value and a run-time
// READ global baked in after 32 calls with no more values than that; each
// moves at most once each way.  So each parameter costs at most
// OP_JIT_MAX_VALUES builds, each once its value is asked for
// OP_JIT_MIN_CALLS times.
class ParamSet {
private:
    // One parameter's statistics under one configuration.
    struct History {
        // The latest value, and the distinct values since it was baked in,
        // or a run-time global's in its current 32 calls, up to one past
        // OP_JIT_MAX_VALUES.
        std::uint64_t last = 0;
        std::vector<std::uint64_t> values;
        std::uint64_t calls = 0;
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
    const Options& m_options;
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

public:
    // The parameters of loop under options, which must outlive them.
    ParamSet(std::string_view loop, const Options& options)
        : m_loop{loop}, m_options{options} {}
    ParamSet(const ParamSet&) = delete;

    // Add a parameter as the generated code describes it, its mode settled
    // by the policy.  Its index.
    std::size_t add(const abi::Param& row) {
        auto& added = m_params.emplace_back(row);
        added.settle(m_loop, m_options);
        m_moving |= !added.fixed();
        return m_params.size() - 1;
    }

    // How the backend lays out the dats a dat_stride parameter reads.
    void set_dat_stride(DatStride stride) { m_dat_stride = stride; }

    std::span<const Param> params() const { return m_params; }
    std::size_t size() const { return m_params.size(); }
    const Param& operator[](std::size_t index) const { return m_params[index]; }

    // The call's values and their hashes, at invoke.
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

    // The key a kernel specialised under a strategy's values, by slot, is
    // built under: each baked parameter's value, hashed 64 at a time with a
    // mask of the run-time ones (a value's hash may be any 64 bits, so no
    // mark could stand in for it).  A strategy's value hashes to its own
    // bits.  The call's statistics are observed first.
    std::uint64_t key(const loop::Call& call, std::span<const int> values) {
        observe_call(call);
        Hasher<std::uint64_t, 65> key{hash_seed};
        std::size_t count = 0;
        std::uint64_t runtime = 0;
        for (const auto& param : m_params) {
            std::uint64_t value = 0;
            if (param.mode() == ParamMode::baked) {
                value = param.source() == abi::ParamSource::strategy
                            ? std::uint64_t{static_cast<std::uint32_t>(
                                  values[static_cast<std::size_t>(param.index())])}
                            : param.hash();
            } else {
                runtime |= std::uint64_t{1} << count;
            }
            key.add(value);
            if (++count == 64) {
                key.add(runtime);
                count = 0;
                runtime = 0;
            }
        }
        return key.add(runtime).value();
    }
};

// The baked parameters as C++ constants, "static constexpr double name =
// value;" a line each, for a compiler that takes source; a strategy's from
// its values, by slot.
std::string format_cxx(const ParamSet& params, std::span<const int> values);

// A parameter's element type in C++: "double".
const char *cxx_type(abi::ElementType type);

} // namespace op::rt::jit
