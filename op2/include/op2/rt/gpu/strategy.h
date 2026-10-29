#pragma once

// The CUDA and HIP strategies: each pairs a kernel with the planner it
// needs and maps its phases onto launches.  The generated code binds each
// launch's kernel arguments through a function it supplies.

#include <op2/rt/gpu/chunk_encode.h>
#include <op2/rt/gpu/chunk_plan.h>
#include <op2/rt/gpu/jit.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/plan/cache.h>
#include <op2/rt/plan/colouring.h>
#include <op_lib_cpp.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace op::rt::gpu {

class Loop;

// What the generated bindings bind a launch's kernel arguments to.
struct LaunchContext {
    int global_stride = 0;
    unsigned opt_flags = 0;
    int start = 0;
    int end = 0;
    int set_stride = 0;
    // A set colouring's elements, colour by colour.
    const int *colour_order = nullptr;

    struct {
        const DevicePlan *plan = nullptr;
        int chunk_begin = 0;
        int chunk_end = 0;
        // This section's position counter under a wavefront partition.
        int *wave_ticket = nullptr;
        int wave_epoch = 0;
    } chunk;
};

// Fill a launch's kernel arguments, one list for its offline and JIT
// kernels alike, from the context.
using Binder = void (*)(LaunchContext& launch, op_arg *args,
                        std::vector<void *>& arguments);

// Facts about a loop that every strategy reads.
struct LoopDescription {
    bool direct;
    // A global OP_INC is incremented atomically, in place, rather than
    // reduced from a copy per thread.
    bool gbl_inc_atomic;
};

// --- Variants -------------------------------------------------------------------

// The block size a strategy runs unless OP_BLOCK_SIZES says otherwise, and
// the offline kernels' launch bounds; a specialised kernel has its own block
// size as launch bounds.
constexpr int default_block_size = 128;

// Every strategy's first axis: OP_<FAMILY>_BLOCK_SIZES, else OP_BLOCK_SIZES.
// The tuner also tries 256 under HIP, and 64 and 256 under CUDA.
inline op::rt::loop::Axis block_size_axis() {
#ifdef OP2_HIP
    return {"BLOCK_SIZES", "block_size", false, {default_block_size, 256}};
#else
    return {"BLOCK_SIZES", "block_size", false, {default_block_size, 64, 256}};
#endif
}

// --- Plans ------------------------------------------------------------------------

// A set colouring and its element order on the device, freed with it.
struct ColouringPlan {
    op::rt::plan::SetColouring colouring;
    int *order_d = nullptr;

    ColouringPlan() = default;
    ColouringPlan(const ColouringPlan&) = delete;
    ColouringPlan& operator=(const ColouringPlan&) = delete;
    ~ColouringPlan() {
        if (order_d != nullptr)
            GPU_CHECK(gpuFree(order_d));
    }
};

// A chunk plan, or why the call does not suit it, and the plan as uploaded.
struct ChunkEntry {
    std::string_view rejected;
    std::unique_ptr<ChunkPlan> plan;
    DevicePlan device;
};

// The live plans of each kind, which every strategy that would build the
// same one shares.
inline op::rt::plan::SharedPlans<ColouringPlan>& colouring_plans() {
    static op::rt::plan::SharedPlans<ColouringPlan> plans;
    return plans;
}

inline op::rt::plan::SharedPlans<ChunkEntry>& chunk_plans() {
    static op::rt::plan::SharedPlans<ChunkEntry> plans;
    return plans;
}

// --- The base of every GPU strategy ------------------------------------------------

class StrategyBase : public op::rt::loop::Strategy {
protected:
    // One launch's geometry.
    struct Geometry {
        int blocks;
        int shared_bytes;
    };

    // The call being prepared: how the sections run, the most blocks a
    // launch may use, and the largest launch.
    struct CallState {
        op::rt::loop::Sections sections;
        int block_limit = 0;
        int max_blocks = 0;
        int global_stride = 0;
    };

    Loop& m_loop;
    std::string m_name;
    KernelSource m_source;
    Binder m_bind;
    int m_block_size;
    // The chunked strategies' switches, constants to every kernel of the
    // loop.
    bool m_unstaged_singles = false;
    bool m_wavefront = false;
    OfflineKernel m_offline;
    op::rt::loop::JitCache<JitKernel> m_cache;
    CallState m_call_state;

    StrategyBase(Loop& loop, const op::rt::loop::Point& point, KernelSource source,
                 Binder bind);

    // This strategy's values for its kernel, from the call being prepared.
    LaunchValues launch_values() const {
        return {m_call_state.global_stride, m_block_size, m_unstaged_singles, m_wavefront};
    }

    // Blocks of a grid-stride launch over elements.
    int grid(int elements) const {
        int blocks = (elements + m_block_size - 1) / m_block_size;
        return std::min(blocks, m_call_state.block_limit);
    }

    // Begin a call: its sections and the blocks the reduction scratch allows.
    void begin(const op::rt::loop::Call& call);

    // Whether its offline kernel launches its block size.
    bool offline_usable() const { return m_offline.usable(m_block_size); }

    // Finish a call: the launch parameters and the kernel of kind, as a
    // Prepared value over entry; a specialised kernel is pending while it
    // builds.
    op::rt::loop::Prepared acquire(const op::rt::loop::Call& call,
                               op::rt::loop::Urgency urgency, op::rt::loop::KernelKind kind,
                               const void *entry);

    // Bind and launch one kernel.
    void launch(const op::rt::loop::Prepared& prepared, LaunchContext& context,
                op_arg *args, Geometry geometry);

    // Run the phases of a call through the loop's, each phase's launches
    // bracketed by an event pair when timed.  launches(phase) gives the
    // launch count, fill(phase, launch, context) the context and geometry.
    template<typename Launches, typename Fill>
    std::optional<float> run_phases(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Launch& launch, int phases,
                                    int wait_before, int globals_after,
                                    Launches&& launches, Fill&& fill);

public:
    static std::vector<op::rt::loop::Axis> axes() { return {block_size_axis()}; }

    std::string_view name() const override { return m_name; }

    // The offline kernel at a nominal 1 when it can launch the block size; a
    // specialised one when there is source, expected faster when the offline
    // kernel uses more than OP_JIT_MIN_REGISTERS registers (by an arbitrary
    // tenth, until timings calibrate it).
    op::rt::loop::Estimate estimate(op::rt::loop::KernelKind kind) const override {
        if (kind == op::rt::loop::KernelKind::offline)
            return offline_usable() ? op::rt::loop::Estimate{true, 1.0f}
                                    : op::rt::loop::Estimate{false, {}, "kernel_unavailable"};
        if (m_source.source.empty())
            return {false, {}, "kernel_unavailable"};
        bool prior = m_offline.registers() > options().min_registers;
        return {true, prior ? 0.9f : 1.0f};
    }

    // The kernel kind and its registers, local and shared memory.
    std::string statistics(const op::rt::loop::Prepared& prepared) const override {
        if (prepared.kernel != nullptr)
            return "jit, " + static_cast<JitKernel *>(prepared.kernel)->figures().text();
        return "offline, " + m_offline.figures().text();
    }

    const CallState& call_state() const { return m_call_state; }

    op::rt::loop::BuildCounts builds() const override { return m_cache.counts(); }

    void release() override { m_cache.join(); }
};

// --- Plain and atomics ------------------------------------------------------------

// One grid-stride launch per section; the family names the kernel's
// increments: none for plain, global atomics for atomics.
class Simple : public StrategyBase {
public:
    Simple(Loop& loop, const op::rt::loop::Point& point, KernelSource source,
           Binder bind)
        : StrategyBase{loop, point, source, bind} {}

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                               op::rt::loop::Urgency urgency,
                               op::rt::loop::KernelKind kind) override {
        begin(call);
        for (const auto& section : m_call_state.sections.span())
            m_call_state.max_blocks = std::max(m_call_state.max_blocks, grid(section.size()));
        return acquire(call, urgency, kind, nullptr);
    }

    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Launch& launch) override;
};

// --- Colouring -----------------------------------------------------------------------

// One launch per colour of a set colouring, as op_plan did under OP_COLOR2.
class Colouring : public StrategyBase {
private:
    op::rt::plan::PlanSlots<ColouringPlan> m_plans;

public:
    Colouring(Loop& loop, const op::rt::loop::Point& point, KernelSource source,
              Binder bind)
        : StrategyBase{loop, point, source, bind} {}

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                               op::rt::loop::Urgency urgency,
                               op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Launch& launch) override;
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;
    void forget(std::uint64_t key) override { m_plans.forget(key); }
    void release() override;
};

// --- The chunked strategies -------------------------------------------------------

// A chunk plan staged in shared memory (op2/rt/gpu/chunk_plan.h): one launch per
// batch of chunks, each block taking chunks in turn.
class ChunkBase : public StrategyBase {
protected:
    bool m_coloured;
    op::rt::plan::ArgGroups m_groups;
    op::rt::plan::PlanSlots<ChunkEntry> m_plans;

    ChunkBase(Loop& loop, const op::rt::loop::Point& point, bool coloured,
              KernelSource source, Binder bind, op::rt::plan::ArgGroups groups)
        : StrategyBase{loop, point, source, bind}, m_coloured{coloured},
          m_groups{groups} {}

    // The block size, which is also the chunk size, and the strategy's
    // switch.
    static std::vector<op::rt::loop::Axis> chunk_axes(op::rt::loop::Axis variant) {
        return {block_size_axis(), std::move(variant)};
    }

    const ChunkEntry& entry_for(const op::rt::loop::Call& call);
    void report(const ChunkResult& result) const;
    int batch_blocks(const ChunkPlan& plan, int begin, int end) const;

public:
    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                               op::rt::loop::Urgency urgency,
                               op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Launch& launch) override;
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;
    void forget(std::uint64_t key) override { m_plans.forget(key); }
    void release() override;

    // Configurations it holds plans of, accepted or rejected.
    std::size_t entries() const { return m_plans.size(); }
};

// Increments staged in shared memory and flushed with global atomics.
class ChunkAtomics : public ChunkBase {
public:
    ChunkAtomics(Loop& loop, const op::rt::loop::Point& point, KernelSource source,
                 Binder bind, op::rt::plan::ArgGroups groups)
        : ChunkBase{loop, point, false, source, bind, groups} {
        m_unstaged_singles = point.value("singles") != 0;
    }

    // OP_CHUNK_ATOMICS_UNSTAGED_SINGLES: a chunk's only reference to a
    // target increments global memory directly, as staging it would save no
    // global atomic.  On by default under HIP, where shared and global fp64
    // atomics are both native; under CUDA a pointer that may be either makes
    // every increment a generic atomic, and shared fp64 adds are
    // compare-and-swap loops.  The HIP tuner also tries it off, which some
    // loops are faster with.
    static std::vector<op::rt::loop::Axis> axes() {
#ifdef OP2_HIP
        return chunk_axes({"UNSTAGED_SINGLES", "singles", true, {1, 0}});
#else
        return chunk_axes({"UNSTAGED_SINGLES", "singles", true, {0}});
#endif
    }
};

// Coloured chunks, then coloured elements within each, so every slot is
// seeded and stored with plain loads and stores.
class ChunkColouring : public ChunkBase {
public:
    ChunkColouring(Loop& loop, const op::rt::loop::Point& point, KernelSource source,
                   Binder bind, op::rt::plan::ArgGroups groups)
        : ChunkBase{loop, point, true, source, bind, groups} {
        m_wavefront = point.value("wavefront") != 0;
    }

    // OP_CHUNK_COLOURING_WAVEFRONT: one launch per section, its chunks taken
    // in an order that keeps them after every earlier chunk sharing a target.
    static std::vector<op::rt::loop::Axis> axes() {
        return chunk_axes({"WAVEFRONT", "wavefront", true, {0}});
    }
};

} // namespace op::rt::gpu
