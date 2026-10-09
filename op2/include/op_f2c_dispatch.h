#pragma once

// Execution strategies for translated Fortran loops, the environment that
// enables and tunes them, and the schedules and launch descriptions a
// selected strategy runs with.

#include <op_hier_plan.h>
#include <op_hier_plan_cache.h>
#include <op_lib_core.h>
#include <op_rt_support.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::f2c {

// Ways to run a loop's conflicting indirect updates, in ladder order: a loop
// runs the first registered strategy that is enabled and accepts it.
enum class Strategy {
    plain,           // no conflicting updates
    hier_atomics,    // increments staged in shared memory, flushed atomically
    atomics,         // increments applied with global atomics
    hier_colouring,  // coloured chunks, then coloured elements within each
    colouring,       // elements coloured across the set by op_plan
};

inline constexpr std::array strategies{
    Strategy::plain,          Strategy::hier_atomics, Strategy::atomics,
    Strategy::hier_colouring, Strategy::colouring,
};
inline constexpr std::size_t strategy_count = strategies.size();

constexpr std::size_t strategy_index(Strategy strategy) {
    return static_cast<std::size_t>(strategy);
}

constexpr std::string_view strategy_name(Strategy strategy) {
    switch (strategy) {
    case Strategy::plain:
        return "plain";
    case Strategy::hier_atomics:
        return "hier_atomics";
    case Strategy::atomics:
        return "atomics";
    case Strategy::hier_colouring:
        return "hier_colouring";
    case Strategy::colouring:
        return "colouring";
    }

    return "unknown";
}

// Return the variable enabling a strategy; plain is never disabled.
constexpr const char *strategy_variable(Strategy strategy) {
    switch (strategy) {
    case Strategy::plain:
        return nullptr;
    case Strategy::hier_atomics:
        return "OP_HIER_ATOMICS";
    case Strategy::atomics:
        return "OP_ATOMICS";
    case Strategy::hier_colouring:
        return "OP_HIER_COLOURING";
    case Strategy::colouring:
        return "OP_COLOURING";
    }

    return nullptr;
}

// Whether a strategy runs over a hierarchical plan.
constexpr bool hierarchical(Strategy strategy) {
    return strategy == Strategy::hier_atomics ||
           strategy == Strategy::hier_colouring;
}

// An enable variable's value: 0, 1, or auto for the default.
enum class StrategySetting {
    off,
    on,
    automatic,
};

// The automatic defaults are per vendor: global atomics on NVIDIA, staged
// atomics on AMD, and hierarchical colouring before op_plan colouring on
// both.
constexpr bool strategy_default(Strategy strategy) {
#if defined(OP2_HIP)
    constexpr bool amd = true;
#else
    constexpr bool amd = false;
#endif

    switch (strategy) {
    case Strategy::hier_atomics:
        return amd;
    case Strategy::atomics:
        return !amd;
    case Strategy::plain:
    case Strategy::hier_colouring:
    case Strategy::colouring:
        return true;
    }

    return false;
}

struct StrategyConfig {
    std::array<StrategySetting, strategy_count> settings{
        StrategySetting::automatic, StrategySetting::automatic,
        StrategySetting::automatic, StrategySetting::automatic,
        StrategySetting::automatic,
    };

    // OP_HIER_ATOMICS_MIN_COMPRESSION: plans averaging fewer staged
    // references per flushed target than this are rejected; 0 accepts all.
    double hier_atomics_min_compression = 1.5;
    // OP_HIER_ATOMICS_EXCLUSIVE=0 keeps every owner on a global atomic, so a
    // run can be compared against one that uses exclusive flushes.
    bool hier_atomics_exclusive = true;
    // OP_HIER_ATOMICS_CHUNK_SIZE and OP_HIER_COLOURING_CHUNK_SIZE; zero
    // leaves the chunk size to OP_part_size and the block size.
    int hier_atomics_chunk_size = 0;
    int hier_colouring_chunk_size = 0;

    bool enabled(Strategy strategy) const {
        switch (settings[strategy_index(strategy)]) {
        case StrategySetting::off:
            return false;
        case StrategySetting::on:
            return true;
        case StrategySetting::automatic:
            return strategy_default(strategy);
        }

        return false;
    }

    int chunk_size(Strategy strategy) const {
        if (strategy == Strategy::hier_atomics)
            return hier_atomics_chunk_size;
        if (strategy == Strategy::hier_colouring)
            return hier_colouring_chunk_size;
        return 0;
    }
};

// Like the JIT settings, parsed once per translation unit from the
// environment.
static StrategyConfig strategy_config;

namespace detail {

inline std::string lower(std::string_view text) {
    std::string out{text};
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

// Parse a boolean variable, warning about anything else.
inline void parse_bool(const char *variable, bool& value) {
    const char *text = std::getenv(variable);
    if (text == nullptr)
        return;

    auto lowered = lower(text);
    if (lowered == "1" || lowered == "yes" || lowered == "true")
        value = true;
    else if (lowered == "0" || lowered == "no" || lowered == "false")
        value = false;
    else
        std::fprintf(stderr, "warning: ignoring %s='%s', expected 0 or 1\n",
                     variable, text);
}

// Parse a non-negative number, warning about anything else.
template<typename T>
void parse_number(const char *variable, T& value) {
    const char *text = std::getenv(variable);
    if (text == nullptr)
        return;

    char *end = nullptr;
    double parsed = std::strtod(text, &end);
    if (end != text && *end == '\0' && parsed >= 0.0)
        value = static_cast<T>(parsed);
    else
        std::fprintf(stderr,
                     "warning: ignoring %s='%s', expected a non-negative "
                     "number\n",
                     variable, text);
}

} // namespace detail

// Read every strategy variable into strategy_config.
static void init_strategy_config() {
    for (Strategy strategy : strategies) {
        const char *variable = strategy_variable(strategy);
        const char *text = variable == nullptr ? nullptr : std::getenv(variable);
        if (text == nullptr)
            continue;

        auto value = detail::lower(text);
        auto& setting = strategy_config.settings[strategy_index(strategy)];
        if (value == "1" || value == "yes" || value == "true") {
            setting = StrategySetting::on;
            std::printf("Enabling %s\n", strategy_name(strategy).data());
        } else if (value == "0" || value == "no" || value == "false") {
            setting = StrategySetting::off;
            std::printf("Disabling %s\n", strategy_name(strategy).data());
        } else if (value != "auto") {
            std::fprintf(stderr,
                         "warning: ignoring %s='%s', expected 0, 1 or auto\n",
                         variable, text);
        }
    }

    detail::parse_number("OP_HIER_ATOMICS_MIN_COMPRESSION",
                         strategy_config.hier_atomics_min_compression);
    detail::parse_bool("OP_HIER_ATOMICS_EXCLUSIVE",
                       strategy_config.hier_atomics_exclusive);
    detail::parse_number("OP_HIER_ATOMICS_CHUNK_SIZE",
                         strategy_config.hier_atomics_chunk_size);
    detail::parse_number("OP_HIER_COLOURING_CHUNK_SIZE",
                         strategy_config.hier_colouring_chunk_size);

    if (OP_diags > 3) {
        std::string ladder;
        for (Strategy strategy : strategies)
            if (strategy_config.enabled(strategy))
                ladder += " " + std::string{strategy_name(strategy)};
        std::printf("strategy: enabled ladder%s\n", ladder.c_str());
    }
}

// Facts about a loop that every strategy reads.
class LoopDescription {
private:
    bool m_direct;
    bool m_gbl_inc_atomic;
    int m_part_size;
    std::vector<int> m_indirect_dats;

    LoopDescription(bool direct, bool gbl_inc_atomic, int part_size,
                    std::vector<int> indirect_dats)
        : m_direct{direct}, m_gbl_inc_atomic{gbl_inc_atomic},
          m_part_size{part_size}, m_indirect_dats{std::move(indirect_dats)} {}

public:
    static LoopDescription direct(bool gbl_inc_atomic) {
        return {true, gbl_inc_atomic, -1, {}};
    }

    // indirect_dats maps each argument to its op_plan indirection index, or
    // -1; part_size is the op_plan block size, or -1 for OP_part_size.
    template<std::size_t N>
    static LoopDescription indirect(const std::array<int, N>& indirect_dats,
                                    int part_size, bool gbl_inc_atomic) {
        return {false, gbl_inc_atomic, part_size,
                {indirect_dats.begin(), indirect_dats.end()}};
    }

    bool is_direct() const { return m_direct; }
    bool gbl_inc_atomic() const { return m_gbl_inc_atomic; }
    int part_size() const { return m_part_size; }
    std::size_t nargs() const { return m_indirect_dats.size(); }
    int ninds() const {
        int max_ind = -1;
        for (int ind : m_indirect_dats)
            max_ind = std::max(max_ind, ind);
        return max_ind + 1;
    }
    int *indirect_dats() { return m_indirect_dats.data(); }
};

class ExecutionSchedule {
private:
    enum class Kind {
        direct,
        atomics,
        colouring,
    };

    Kind m_kind;
    op_set m_set = nullptr;
    op_plan *m_plan = nullptr;
    bool m_separate_owned = false;

    ExecutionSchedule(Kind kind, op_set set, op_plan *plan,
                      bool separate_owned)
        : m_kind{kind}, m_set{set}, m_plan{plan},
          m_separate_owned{separate_owned} {}

public:
    static ExecutionSchedule direct(op_set set) {
        return ExecutionSchedule{Kind::direct, set, nullptr, false};
    }

    static ExecutionSchedule atomics(op_set set, bool separate_owned) {
        return ExecutionSchedule{Kind::atomics, set, nullptr, separate_owned};
    }

    static ExecutionSchedule colouring(op_set set, op_plan *plan) {
        return ExecutionSchedule{Kind::colouring, set, plan, false};
    }

    int size() const {
        switch (m_kind) {
        case Kind::direct:
            return 1;
        case Kind::atomics:
            return m_separate_owned ? 3 : 2;
        case Kind::colouring:
            return m_plan->ncolors;
        }

        assert(false);
        return 0;
    }

    ExecutionSection operator[](int index) const {
        assert(index >= 0 && index < size());

        switch (m_kind) {
        case Kind::direct:
            return {0, static_cast<int>(m_set->size)};
        case Kind::atomics:
            if (index == 0)
                return {0, m_set->core_size};
            if (m_separate_owned && index == 1)
                return {m_set->core_size, static_cast<int>(m_set->size)};

            return {m_separate_owned ? static_cast<int>(m_set->size)
                                     : m_set->core_size,
                    static_cast<int>(m_set->size) + m_set->exec_size};
        case Kind::colouring:
            return {m_plan->col_offsets[0][index],
                    m_plan->col_offsets[0][index + 1]};
        }

        assert(false);
        return {0, 0};
    }

    // Every exchange is paired with exactly one wait, so exactly one index in
    // [0, size()] answers true; size() means after the last section, which is
    // where a colouring plan with no non-core colour waits.
    bool wait_before(int index) const {
        assert(index >= 0 && index <= size());

        switch (m_kind) {
        case Kind::direct:
            // A direct loop reads no halo: its exchange only brings the dats
            // to this device.
            return index == 0;
        case Kind::atomics:
            return index == 1;
        case Kind::colouring:
            return index == m_plan->ncolors_core;
        }

        assert(false);
        return false;
    }

    bool process_globals_after(int index) const {
        assert(index >= 0 && index < size());

        switch (m_kind) {
        case Kind::direct:
            return index == 0;
        case Kind::atomics:
            return index == 1;
        case Kind::colouring:
            return index == m_plan->ncolors_owned - 1;
        }

        assert(false);
        return false;
    }

    int set_stride() const {
        int size = m_set->size;
        if (m_kind != Kind::direct)
            size += m_set->exec_size;

        return (size + 31) & ~31;
    }

    int *color_reorder() const {
        return m_kind == Kind::colouring ? m_plan->col_reord : nullptr;
    }
};

struct KernelExecutionOptions {
    // Start the ladder at this strategy, enabled whatever the environment
    // says.  A forced strategy registered without argument groups runs
    // without a plan, with shared_bytes of dynamic shared memory.
    std::optional<Strategy> force;
    int shared_bytes = 0;
};

class JitKernel;

struct KernelExecution {
    Strategy strategy;
    JitKernel *jit_kernel;
    ExecutionSchedule schedule;
    int block_size;
    int block_limit;
    int max_blocks;
    int shared_bytes;
    const HierPlan *plan;
    HierPlanDeviceView plan_device;
    // Why each strategy above the selected one was passed over.
    std::array<FallbackReason, strategy_count> skipped;

    // Return how many launches run a section: one per block colour under a
    // colouring plan, otherwise one.
    int launches(int section_index) const {
        if (plan == nullptr || !plan->colouring)
            return 1;

        const auto& offsets = plan->colouring->section_launch_offsets;
        return offsets[section_index + 1] - offsets[section_index];
    }

    // Return the plan chunks one launch covers; under a colouring plan they
    // index chunk_order.
    std::pair<int, int> chunk_range(int section_index, int launch) const {
        assert(plan != nullptr);
        if (!plan->colouring)
            return {plan->section_chunk_offsets[section_index],
                    plan->section_chunk_offsets[section_index + 1]};

        const auto& colouring = *plan->colouring;
        int index = colouring.section_launch_offsets[section_index] + launch;
        return {colouring.launch_chunk_offsets[index],
                colouring.launch_chunk_offsets[index + 1]};
    }

    int num_blocks(int section_index, int launch) const {
        if (plan != nullptr) {
            auto [begin, end] = chunk_range(section_index, launch);
            return std::min(end - begin, block_limit);
        }

        auto section = schedule[section_index];
        int blocks = (section.size() + block_size - 1) / block_size;
        return std::min(blocks, block_limit);
    }

    int dynamic_shared_bytes(int section_index) const {
        if (plan == nullptr)
            return shared_bytes;
        if (!plan->staging)
            return 0;

        auto bytes = plan->staging->section_shared_bytes[section_index];
        assert(bytes <= static_cast<std::size_t>(INT32_MAX));
        return static_cast<int>(bytes);
    }
};

struct LaunchContext {
    int global_stride;
    unsigned opt_flags;
    int start;
    int end;
    int set_stride;
    int *color_reorder;

    struct {
        HierPlanDeviceView plan;
        int chunk_begin;
        int chunk_end;
        int has_exclusive;
    } hier;
};

struct GlobalInitContext {
    int block_size;
    int max_blocks;
    int global_stride;
};

template<std::size_t OfflineN, std::size_t JitN>
struct KernelArguments {
    std::array<void *, OfflineN> offline;
    std::array<void *, JitN> jit;
};

template<std::size_t OfflineN, std::size_t JitN>
KernelArguments(std::array<void *, OfflineN>, std::array<void *, JitN>)
    -> KernelArguments<OfflineN, JitN>;

struct KernelInvocationResult {
    bool used_jit = false;
    Strategy strategy = Strategy::plain;
    int block_size = 0;
    int max_blocks = 0;
    std::array<FallbackReason, strategy_count> skipped{};

    // Return why a strategy above the selected one was passed over.
    FallbackReason skipped_reason(Strategy strategy) const {
        return skipped[strategy_index(strategy)];
    }
};

} // namespace op::f2c
