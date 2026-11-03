#pragma once

// The CUDA and HIP strategies: each pairs a family's kernel with the planner
// it needs and maps its phases onto launches, each given a launch record
// (op2/rt/abi/gpu/launch.h).  The generated code binds the loop's arguments
// through a function it supplies, and launches an offline kernel through the
// calls it supplies.  Defined in src/rt/gpu/op_gpu_strategy.cpp.

#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/abi/gpu/launch.h>
#include <op2/rt/gpu/chunk_encode.h>
#include <op2/rt/gpu/chunk_plan.h>
#include <op2/rt/gpu/executor.h>
#include <op2/rt/gpu/jit.h>
#include <op2/rt/gpu/layout.h>
#include <op2/rt/gpu/options.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/plan/cache.h>
#include <op2/rt/plan/colouring.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::gpu {

// --- Variants -------------------------------------------------------------------

// Every strategy's first axis: OP_<FAMILY>_BLOCK_SIZES, else OP_BLOCK_SIZES.
// The tuner also tries 256 under HIP, and 64 and 256 under CUDA.
op::rt::loop::Axis block_size_axis();

// --- Plans ------------------------------------------------------------------------

// A set colouring and its element order on the device, freed with it, and
// its largest colour's elements.
struct ColouringPlan {
    op::rt::plan::SetColouring colouring;
    int *order_d = nullptr;
    int max_colour = 0;

    ColouringPlan() = default;
    ColouringPlan(const ColouringPlan&) = delete;
    ColouringPlan& operator=(const ColouringPlan&) = delete;
    ~ColouringPlan();
};

// A chunk plan, or why the call does not suit it, the plan as uploaded, and
// the most chunks a batch launches.
struct ChunkEntry : ChunkResult {
    DevicePlan device;
    int max_batch = 0;
};

// --- The base of every GPU strategy ------------------------------------------------

class StrategyBase : public op::rt::loop::Strategy {
protected:
    // One launch's geometry.
    struct Geometry {
        int blocks;
        int shared_bytes;
    };

    // A call's launches under its plan: the per-element arrays' stride, the
    // most blocks a launch may use, and the stride between the blocks' copies
    // of the globals, its block size times the largest launch.  Derived from
    // the call and the plan wherever it is needed, so prepare, begin_call and
    // run agree.
    struct Shape {
        int stride = 0;
        int block_limit = 0;
        int global_stride = 0;
    };

    // A call's plan, or why the configuration does not suit it.
    struct Planned {
        const void *entry = nullptr;
        std::string_view rejected;
    };

    Executor& m_executor;
    std::string m_name;
    const abi::gpu::Family& m_family;
    int m_block_size;
    OfflineKernel m_offline;
    op::rt::jit::Cache<JitKernel> m_cache;

    StrategyBase(Executor& executor, const op::rt::loop::Point& point,
                 const abi::gpu::Family& family);

    // The configuration's plan for call, its per-element arrays stride
    // apart.
    virtual Planned plan(const op::rt::loop::Call& call, int stride) = 0;

    // The largest launch's blocks for call under its plan's entry, at most
    // block_limit.
    virtual int max_blocks(const op::rt::loop::Call& call, const void *entry,
                           int block_limit) const = 0;

    // This strategy's values for its kernel under global_stride; a chunked
    // variant adds its switch.
    virtual LaunchValues launch_values(int global_stride) const {
        return {global_stride, m_block_size, false, false};
    }

    // Blocks of a grid-stride launch over elements, at most block_limit.
    int grid(int elements, int block_limit) const {
        int blocks = (elements + m_block_size - 1) / m_block_size;
        return std::min(blocks, block_limit);
    }

    Shape shape(const op::rt::loop::Call& call, const void *entry) const;

    // Whether its offline kernel launches its block size.
    bool offline_usable() const { return m_offline.usable(m_block_size); }

    // Launch one kernel with record.
    template<typename Record>
    void launch(const op::rt::loop::Prepared& prepared, Record& record,
                std::span<op_arg> args, Geometry geometry);

    // Run the phases of a call through the loop's, each phase's launches
    // bracketed by an event pair when timed.  launches(phase) gives the
    // launch count, fill(phase, launch, record) the launch's record of type
    // Record past its strides, and its geometry.
    template<typename Record, typename Launches, typename Fill>
    std::optional<float> run_phases(const op::rt::loop::Prepared& prepared,
                                    const op::rt::loop::Call& call, bool timed,
                                    const op::rt::loop::Phases& phases, const Shape& shape,
                                    Launches&& launches, Fill&& fill);

public:
    static std::vector<op::rt::loop::Axis> axes() { return {block_size_axis()}; }

    std::string_view name() const override { return m_name; }

    // Its kernels as the backend's heuristics estimate them
    // (op2/rt/gpu/heuristics.h): the offline one when it can launch the block
    // size, and a specialised one when there is source.
    op::rt::loop::Estimate estimate(op::rt::loop::KernelKind kind) const override;

    // The kernel kind and its registers, local and shared memory.
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;

    // Plan the call, then find or request its kernel of kind, as urgency
    // asks: an offline one is ready with the plan, a specialised one pending
    // while it builds.
    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                                   op::rt::loop::Urgency urgency,
                                   op::rt::loop::KernelKind kind) final;

    // The device copies of the call's globals, a copy per launched thread.
    void begin_call(const op::rt::loop::Prepared& prepared,
                    const op::rt::loop::Call& call) override {
        m_executor.prepare_globals(prepared, call, shape(call, prepared.entry).global_stride);
    }

    op::rt::loop::BuildCounts builds() const override { return m_cache.counts(); }
};

// --- Plain and atomics ------------------------------------------------------------

// One grid-stride launch per section; the family names the kernel's
// increments: none for plain, global atomics for atomics.
class Simple : public StrategyBase {
protected:
    Planned plan(const op::rt::loop::Call&, int) override { return {}; }
    int max_blocks(const op::rt::loop::Call& call, const void *entry,
                   int block_limit) const override;

public:
    Simple(Executor& executor, const op::rt::loop::Point& point, const abi::gpu::Family& family)
        : StrategyBase{executor, point, family} {}

    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Call& call,
                             bool timed) override;
};

// --- Colouring -----------------------------------------------------------------------

// One launch per colour of a set colouring, as op_plan did under OP_COLOR2.
class Colouring : public StrategyBase {
private:
    op::rt::plan::PlanSlots<ColouringPlan> m_plans;

protected:
    Planned plan(const op::rt::loop::Call& call, int stride) override;
    int max_blocks(const op::rt::loop::Call& call, const void *entry,
                   int block_limit) const override;

public:
    Colouring(Executor& executor, const op::rt::loop::Point& point, const abi::gpu::Family& family)
        : StrategyBase{executor, point, family} {}

    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Call& call,
                             bool timed) override;
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;
    void forget(std::uint64_t key) override { m_plans.forget(key); }
};

// --- The chunked strategies -------------------------------------------------------

// A chunk plan staged in shared memory (op2/rt/gpu/chunk_plan.h): one launch per
// batch of chunks, each block taking chunks in turn.
class ChunkBase : public StrategyBase {
protected:
    op::rt::abi::ArgGroups m_groups;
    op::rt::plan::PlanSlots<ChunkEntry> m_plans;

    ChunkBase(Executor& executor, const op::rt::loop::Point& point, const abi::gpu::Family& family)
        : StrategyBase{executor, point, family}, m_groups{family.groups} {}

    // The variant's plan of the call under spec, its inputs to the plan's key
    // beyond the shared ones, and its own figures of a plan.
    virtual ChunkResult make_plan(const op::rt::loop::Call& call, const ChunkSpec& spec) const = 0;
    virtual void add_key(op::rt::plan::PlanKey& key) const = 0;
    virtual std::string figures(const ChunkPlan& plan) const = 0;

    // The block size, which is also the chunk size, and the strategy's
    // switch.
    static std::vector<op::rt::loop::Axis> chunk_axes(op::rt::loop::Axis variant) {
        return {block_size_axis(), std::move(variant)};
    }

    void report(const ChunkResult& result) const;
    // A batch's blocks: one per chunk up to block_limit, and for a wavefront
    // no more than the device holds at once.
    int batch_blocks(const ChunkPlan& plan, int chunks, int block_limit) const;

    Planned plan(const op::rt::loop::Call& call, int stride) override;
    int max_blocks(const op::rt::loop::Call& call, const void *entry,
                   int block_limit) const override;

public:
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Call& call,
                             bool timed) override;
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;
    void forget(std::uint64_t key) override { m_plans.forget(key); }
};

// Increments staged in shared memory and flushed with global atomics.
class ChunkAtomics : public ChunkBase {
protected:
    bool m_unstaged_singles;

    LaunchValues launch_values(int global_stride) const override {
        auto values = ChunkBase::launch_values(global_stride);
        values.unstaged_singles = m_unstaged_singles;
        return values;
    }
    ChunkResult make_plan(const op::rt::loop::Call& call, const ChunkSpec& spec) const override;
    void add_key(op::rt::plan::PlanKey& key) const override;
    std::string figures(const ChunkPlan& plan) const override;

public:
    ChunkAtomics(Executor& executor, const op::rt::loop::Point& point,
                 const abi::gpu::Family& family)
        : ChunkBase{executor, point, family}, m_unstaged_singles{point.value("singles") != 0} {}

    // OP_CHUNK_ATOMICS_UNSTAGED_SINGLES: a chunk's only reference to a
    // target increments global memory directly, as staging it would save no
    // global atomic.  On by default under HIP, where shared and global fp64
    // atomics are both native; under CUDA a pointer that may be either makes
    // every increment a generic atomic, and shared fp64 adds are
    // compare-and-swap loops.  The HIP tuner also tries it off, which some
    // loops are faster with.
    static std::vector<op::rt::loop::Axis> axes();
};

// Coloured chunks, then coloured elements within each, so every slot is
// seeded and stored with plain loads and stores.
class ChunkColouring : public ChunkBase {
protected:
    bool m_wavefront;

    LaunchValues launch_values(int global_stride) const override {
        auto values = ChunkBase::launch_values(global_stride);
        values.wavefront = m_wavefront;
        return values;
    }
    ChunkResult make_plan(const op::rt::loop::Call& call, const ChunkSpec& spec) const override;
    void add_key(op::rt::plan::PlanKey& key) const override;
    std::string figures(const ChunkPlan& plan) const override;

public:
    ChunkColouring(Executor& executor, const op::rt::loop::Point& point,
                   const abi::gpu::Family& family)
        : ChunkBase{executor, point, family}, m_wavefront{point.value("wavefront") != 0} {}

    // OP_CHUNK_COLOURING_WAVEFRONT: one launch per section, its chunks taken
    // in an order that keeps them after every earlier chunk sharing a target.
    static std::vector<op::rt::loop::Axis> axes() {
        return chunk_axes({"WAVEFRONT", "wavefront", true, {0}});
    }
};

// --- Families -------------------------------------------------------------------

// A strategy family as the GPU backend offers it.
using OfferedFamily = op::rt::loop::OfferedFamily<Executor, abi::gpu::Family>;

// The families in ladder order: plain, chunk_atomics (on by default under
// HIP, where its unstaged singles are native atomics), atomics,
// chunk_colouring and colouring.
std::span<const OfferedFamily> strategy_families();

} // namespace op::rt::gpu
