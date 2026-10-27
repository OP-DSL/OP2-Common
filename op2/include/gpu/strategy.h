#pragma once

// The CUDA and HIP strategies: each pairs a kernel with the planner it
// needs and maps its phases onto launches.  The generated code binds each
// launch's kernel arguments through a function it supplies.

#include <gpu/chunk_encode.h>
#include <gpu/chunk_plan.h>
#include <gpu/jit.h>
#include <loop/strategy.h>
#include <op_lib_cpp.h>
#include <plan/colouring.h>
#include <plan/options.h>

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

namespace op::gpu {

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
// the HIP offline kernels' launch bound.
constexpr int default_block_size = 128;

// Every strategy's first axis: OP_<FAMILY>_BLOCK_SIZES, else OP_BLOCK_SIZES.
inline op::loop::Axis block_size_axis() {
    return {"BLOCK_SIZES", "block_size", false, {default_block_size}};
}

// Whether the offline kernels have launch bounds: under HIP, whose compiler
// otherwise allows for 1024 threads a block.
#ifdef OP2_HIP
constexpr bool offline_launch_bounds = true;
#else
constexpr bool offline_launch_bounds = false;
#endif

// OP_<FAMILY>_LAUNCH_BOUNDS, else OP_LAUNCH_BOUNDS: the specialised kernel
// has its block size as launch bounds, which the compiler also takes as an
// occupancy target and may cut registers for.  As the offline kernels by
// default.
inline op::loop::Axis launch_bounds_axis() {
    return {"LAUNCH_BOUNDS", "launch_bounds", true, {offline_launch_bounds}};
}

// OP_<FAMILY>_SMEM_SPILLING, else OP_SMEM_SPILLING (CUDA 13 and later, off
// by default): the specialised kernel spills registers to shared memory
// before local memory.  It needs launch bounds, which size the shared memory
// it takes, and a kernel without dynamic shared memory, so the chunked
// strategies have no such axis.
#if defined(OP2_CUDA) && CUDART_VERSION >= 13000
constexpr bool smem_spilling_supported = true;
#else
constexpr bool smem_spilling_supported = false;
#endif

inline op::loop::Axis smem_spilling_axis() {
    return {"SMEM_SPILLING", "smem_spilling", true, {0}};
}

// --- The base of every GPU strategy ------------------------------------------------

class StrategyBase : public op::loop::Strategy {
protected:
    // One launch's geometry.
    struct Geometry {
        int blocks;
        int shared_bytes;
    };

    // The call being prepared: how the sections run, the most blocks a
    // launch may use, and the largest launch.
    struct CallState {
        op::loop::Sections sections;
        int block_limit = 0;
        int max_blocks = 0;
        int global_stride = 0;
    };

    Loop& m_loop;
    std::string m_name;
    KernelSource m_source;
    Binder m_bind;
    int m_block_size;
    // How its specialised kernel is compiled.
    CompileSwitches m_switches;
    // The chunked strategies' switches, constants to every kernel of the
    // loop.
    bool m_unstaged_singles = false;
    bool m_wavefront = false;
    OfflineKernel m_offline;
    op::loop::JitCache<JitKernel> m_cache;
    CallState m_call_state;

    StrategyBase(Loop& loop, const op::loop::Point& point, KernelSource source,
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
    void begin(const op::loop::Call& call);

    // Whether its offline kernel launches its block size and stands for it:
    // compiled with its switches.
    bool offline_usable() const {
        return m_offline.usable(m_block_size) &&
               m_switches.launch_bounds == offline_launch_bounds &&
               !m_switches.smem_spilling;
    }

    // Whether no kernel of kind can come about: the offline kind with no
    // usable offline kernel.  A strategy that plans checks this first.
    bool never_available(op::loop::KernelKind kind) const {
        return kind == op::loop::KernelKind::offline && !offline_usable();
    }

    op::loop::Prepared unavailable() {
        op::loop::Prepared prepared;
        prepared.status = op::loop::Status::unavailable;
        prepared.reason = "kernel_unavailable";
        return prepared;
    }

    // Finish a call: the launch parameters and the kernel of kind, as a
    // Prepared value over entry; a specialised kernel is pending while it
    // builds.
    op::loop::Prepared acquire(const op::loop::Call& call,
                               op::loop::Urgency urgency, op::loop::KernelKind kind,
                               const void *entry);

    // Bind and launch one kernel.
    void launch(const op::loop::Prepared& prepared, LaunchContext& context,
                op_arg *args, Geometry geometry);

    // Run the phases of a call: before phase wait_before the halo wait, after
    // phase globals_after the global processing, each phase's launches
    // bracketed by an event pair when timed.  launches(phase) gives the
    // launch count, fill(phase, launch, context) the context and geometry.
    template<typename Launches, typename Fill>
    std::optional<float> run_phases(const op::loop::Prepared& prepared,
                                    const op::loop::Launch& launch, int phases,
                                    int wait_before, int globals_after,
                                    Launches&& launches, Fill&& fill);

public:
    static std::vector<op::loop::Axis> axes() {
        std::vector<op::loop::Axis> out{block_size_axis(), launch_bounds_axis()};
        if (smem_spilling_supported)
            out.push_back(smem_spilling_axis());
        return out;
    }

    std::string_view name() const override { return m_name; }

    // The offline kernel at a nominal 1 when it can launch the block size
    // and has the switches; a specialised one when there is source, expected
    // faster when the offline kernel uses more than OP_JIT_MIN_REGISTERS
    // registers (by an arbitrary tenth, until timings calibrate it).
    op::loop::Estimate estimate(op::loop::KernelKind kind) const override {
        if (kind == op::loop::KernelKind::offline)
            return {offline_usable(), 1.0f};
        bool prior = m_offline.registers() > options().min_registers;
        return {!m_source.source.empty(), prior ? 0.9f : 1.0f};
    }

    // The kernel kind and its registers, local and shared memory.
    std::string statistics(const op::loop::Prepared& prepared) const override {
        if (prepared.kernel != nullptr)
            return "jit, " + static_cast<JitKernel *>(prepared.kernel)->figures().text();
        return "offline, " + m_offline.figures().text();
    }

    const CallState& call_state() const { return m_call_state; }

    op::loop::BuildCounts builds() const { return m_cache.counts(); }

    // Free device state while the device is up.
    virtual void release() { m_cache.join(); }
};

// --- Plain and atomics ------------------------------------------------------------

// One grid-stride launch per section; the family names the kernel's
// increments: none for plain, global atomics for atomics.
class Simple : public StrategyBase {
public:
    Simple(Loop& loop, const op::loop::Point& point, KernelSource source,
           Binder bind)
        : StrategyBase{loop, point, source, bind} {}

    op::loop::Prepared prepare(const op::loop::Call& call,
                               op::loop::Urgency urgency,
                               op::loop::KernelKind kind) override {
        begin(call);
        for (const auto& section : m_call_state.sections.span())
            m_call_state.max_blocks = std::max(m_call_state.max_blocks, grid(section.size()));
        return acquire(call, urgency, kind, nullptr);
    }

    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
};

// --- Colouring -----------------------------------------------------------------------

// One launch per colour of a set colouring, as op_plan did under OP_COLOR2.
class Colouring : public StrategyBase {
private:
    struct Entry {
        op::plan::SetColouring colouring;
        int *order_d = nullptr;
    };

    std::unordered_map<std::uint64_t, Entry> m_entries;

public:
    Colouring(Loop& loop, const op::loop::Point& point, KernelSource source,
              Binder bind)
        : StrategyBase{loop, point, source, bind} {}

    op::loop::Prepared prepare(const op::loop::Call& call,
                               op::loop::Urgency urgency,
                               op::loop::KernelKind kind) override;
    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
    std::string statistics(const op::loop::Prepared& prepared) const override;
    void release() override;
};

// --- The chunked strategies -------------------------------------------------------

// A chunk plan staged in shared memory (gpu/chunk_plan.h): one launch per
// batch of chunks, each block taking chunks in turn.
class ChunkBase : public StrategyBase {
protected:
    struct Entry {
        std::string_view rejected;
        std::unique_ptr<ChunkPlan> plan;
        DevicePlan device;
    };

    bool m_coloured;
    op::plan::ArgGroups m_groups;
    int m_chunk_size;
    std::unordered_map<std::uint64_t, Entry> m_entries;

    ChunkBase(Loop& loop, const op::loop::Point& point, bool coloured,
              KernelSource source, Binder bind, op::plan::ArgGroups groups)
        : StrategyBase{loop, point, source, bind}, m_coloured{coloured},
          m_groups{groups}, m_chunk_size{point.value("chunk_size")} {}

    // The block size, launch bounds, the chunk size, a block multiple of 128
    // elements by default, and the strategy's switch.
    static std::vector<op::loop::Axis> chunk_axes(op::loop::Axis variant) {
        return {block_size_axis(), launch_bounds_axis(),
                op::plan::chunk_size_axis(default_block_size), std::move(variant)};
    }

    Entry& entry_for(const op::loop::Call& call);
    void report(const ChunkResult& result) const;
    int batch_blocks(const ChunkPlan& plan, int begin, int end) const;

public:
    op::loop::Prepared prepare(const op::loop::Call& call,
                               op::loop::Urgency urgency,
                               op::loop::KernelKind kind) override;
    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
    std::string statistics(const op::loop::Prepared& prepared) const override;
    void release() override;

    // Configurations planned so far, accepted or rejected.
    std::size_t entries() const { return m_entries.size(); }
};

// Increments staged in shared memory and flushed with global atomics.
class ChunkAtomics : public ChunkBase {
public:
    ChunkAtomics(Loop& loop, const op::loop::Point& point, KernelSource source,
                 Binder bind, op::plan::ArgGroups groups)
        : ChunkBase{loop, point, false, source, bind, groups} {
        m_unstaged_singles = point.value("singles") != 0;
    }

    // OP_CHUNK_ATOMICS_UNSTAGED_SINGLES: a chunk's only reference to a
    // target increments global memory directly, as staging it would save no
    // global atomic.  On by default under HIP, where shared and global fp64
    // atomics are both native; under CUDA a pointer that may be either makes
    // every increment a generic atomic, and shared fp64 adds are
    // compare-and-swap loops.
    static std::vector<op::loop::Axis> axes() {
#ifdef OP2_HIP
        constexpr int unstaged_singles = 1;
#else
        constexpr int unstaged_singles = 0;
#endif
        return chunk_axes({"UNSTAGED_SINGLES", "singles", true, {unstaged_singles}});
    }
};

// Coloured chunks, then coloured elements within each, so every slot is
// seeded and stored with plain loads and stores.
class ChunkColouring : public ChunkBase {
public:
    ChunkColouring(Loop& loop, const op::loop::Point& point, KernelSource source,
                   Binder bind, op::plan::ArgGroups groups)
        : ChunkBase{loop, point, true, source, bind, groups} {
        m_wavefront = point.value("wavefront") != 0;
    }

    // OP_CHUNK_COLOURING_WAVEFRONT: one launch per section, its chunks taken
    // in an order that keeps them after every earlier chunk sharing a target.
    static std::vector<op::loop::Axis> axes() {
        return chunk_axes({"WAVEFRONT", "wavefront", true, {0}});
    }
};

} // namespace op::gpu
