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

extern "C" {
int getBlockLimitWithPolicy(op_arg *args, int nargs, int block_size,
                            const char *name, bool gbl_inc_atomic);
void prepareDeviceGblsWithPolicy(op_arg *args, int nargs, int max_threads,
                                 bool gbl_inc_atomic);
bool processDeviceGblsWithPolicy(op_arg *args, int nargs, int nelems,
                                 int max_threads, bool gbl_inc_atomic);
}

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

// Fill a launch's kernel arguments, offline and JIT, from the context.
using Binder = void (*)(LaunchContext& launch, op_arg *args,
                        std::vector<void *>& offline, std::vector<void *>& jit);

// Facts about a loop that every strategy reads.
struct LoopDescription {
    bool direct;
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
    struct Call {
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
    // The chunked strategies' switches, constants to every kernel of the
    // loop.
    bool m_unstaged_singles = false;
    bool m_wavefront = false;
    JitKernels m_kernels;
    Call m_call;

    StrategyBase(Loop& loop, const op::loop::Point& point, KernelSource source,
                 Binder bind);

    // This strategy's values for its kernel, from the call being prepared.
    LaunchValues launch_values() const {
        return {m_call.global_stride, m_block_size, m_unstaged_singles, m_wavefront};
    }

    // Blocks of a grid-stride launch over elements.
    int grid(int elements) const {
        int blocks = (elements + m_block_size - 1) / m_block_size;
        return std::min(blocks, m_call.block_limit);
    }

    // Begin a call: its sections and the blocks the reduction scratch allows.
    void begin(const op::loop::Configuration& configuration);

    // Whether no kernel can ever come about: no usable offline kernel, and
    // the JIT off.  A strategy that plans checks this before planning.
    bool never_available() const {
        return !config().enable &&
               !JitService::offline_usable(m_kernels, m_source, m_block_size);
    }

    op::loop::Prepared unavailable() {
        op::loop::Prepared prepared;
        prepared.strategy = this;
        prepared.status = op::loop::Status::unavailable;
        prepared.reason = "kernel_unavailable";
        return prepared;
    }

    // Finish a call: the launch parameters and the kernel, as a Prepared
    // value over entry.
    op::loop::Prepared acquire(op::loop::Urgency urgency, const void *entry);

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
    static std::vector<op::loop::Axis> axes() { return {block_size_axis()}; }

    std::string_view name() const override { return m_name; }

    std::string statistics(const op::loop::Prepared& prepared) const override {
        return prepared.kernel != nullptr ? "jit" : "offline";
    }

    const Call& call() const { return m_call; }
    int block_size() const { return m_block_size; }

    // Free device state while the device is up.
    virtual void release() { m_kernels.join(); }
};

// --- Plain and atomics ------------------------------------------------------------

// One grid-stride launch per section; the family names the kernel's
// increments: none for plain, global atomics for atomics.
class Simple : public StrategyBase {
public:
    Simple(Loop& loop, const op::loop::Point& point, KernelSource source,
           Binder bind)
        : StrategyBase{loop, point, source, bind} {}

    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               op::loop::Urgency urgency) override {
        begin(configuration);
        for (const auto& section : m_call.sections.span())
            m_call.max_blocks = std::max(m_call.max_blocks, grid(section.size()));
        return acquire(urgency, nullptr);
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

    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               op::loop::Urgency urgency) override;
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
          m_groups{groups}, m_chunk_size{point.values[1]} {}

    // The block size, the chunk size, a block multiple of 128 elements by
    // default, and the strategy's switch.
    static std::vector<op::loop::Axis> chunk_axes(op::loop::Axis variant) {
        return {block_size_axis(), op::plan::chunk_size_axis(default_block_size),
                std::move(variant)};
    }

    Entry& entry_for(const op::loop::Configuration& configuration);
    void report(const ChunkResult& result) const;

public:
    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               op::loop::Urgency urgency) override;
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
        m_unstaged_singles = point.values[2] != 0;
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
        m_wavefront = point.values[2] != 0;
    }

    // OP_CHUNK_COLOURING_WAVEFRONT: one launch per section, its chunks taken
    // in an order that keeps them after every earlier chunk sharing a target.
    static std::vector<op::loop::Axis> axes() {
        return chunk_axes({"WAVEFRONT", "wavefront", true, {0}});
    }
};

} // namespace op::gpu
