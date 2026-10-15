#pragma once

// The CUDA and HIP strategies: each pairs a kernel with the planner it
// needs and maps its phases onto launches.  The generated code binds each
// launch's kernel arguments through a function it supplies.

#include <gpu/chunk_encode.h>
#include <loop/chunk_partition.h>
#include <gpu/jit.h>
#include <op_lib_cpp.h>
#include <loop/strategy.h>
#include <loop/set_colouring.h>

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

struct GlobalsContext {
    int block_size;
    int max_blocks;
    int global_stride;
};

// Initialise the per-thread global reduction storage before a loop runs.
using GlobalsInit = void (*)(const GlobalsContext& context, op_arg *args);

// Facts about a loop that every strategy reads.
struct LoopDescription {
    bool direct;
    bool gbl_inc_atomic;
};

// --- The base of every GPU strategy ------------------------------------------------

class StrategyBase : public op::loop::Strategy {
protected:
    // One launch's geometry.
    struct Geometry {
        int blocks;
        int shared_bytes;
    };

    // The call being prepared: how the sections run, the most blocks a
    // launch may use, the largest launch, and the kernel.
    struct Call {
        op::loop::Sections sections;
        int block_limit = 0;
        int max_blocks = 0;
        int global_stride = 0;
        JitKernel *kernel = nullptr;
    };

    Loop& m_loop;
    std::string m_name;
    KernelSource m_source;
    Binder m_bind;
    int m_block_size;
    JitKernels m_kernels;
    Call m_call;

    StrategyBase(Loop& loop, std::string_view family, KernelSource source,
                 Binder bind, int block_size);

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
    op::loop::Prepared acquire(bool now, const void *entry);

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
    Simple(Loop& loop, std::string_view family, KernelSource source,
           Binder bind, int block_size)
        : StrategyBase{loop, family, source, bind, block_size} {}

    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               bool now) override {
        begin(configuration);
        for (const auto& section : m_call.sections.span())
            m_call.max_blocks = std::max(m_call.max_blocks, grid(section.size()));
        return acquire(now, nullptr);
    }

    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
};

// --- Colouring -----------------------------------------------------------------------

// One launch per colour of a set colouring, as op_plan did under OP_COLOR2.
class Colouring : public StrategyBase {
private:
    struct Entry {
        op::colouring::SetColouring colouring;
        int *order_d = nullptr;
    };

    std::unordered_map<std::uint64_t, Entry> m_entries;
    Entry *m_entry = nullptr;

public:
    Colouring(Loop& loop, KernelSource source, Binder bind, int block_size)
        : StrategyBase{loop, "colouring", source, bind, block_size} {}

    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               bool now) override;
    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
    std::string statistics(const op::loop::Prepared& prepared) const override;
    void release() override;
};

// --- The chunked strategies -------------------------------------------------------

// A chunk partition staged in shared memory: one launch per batch of
// chunks, each block taking chunks in turn.
class ChunkBase : public StrategyBase {
protected:
    struct Entry {
        std::string_view rejected;
        std::unique_ptr<op::chunk::Partition> partition;
        DevicePlan plan;
    };

    op::chunk::Kind m_kind;
    op::chunk::ArgGroups m_groups;
    int m_chunk_size;
    std::unordered_map<std::uint64_t, Entry> m_entries;
    Entry *m_entry = nullptr;
    int m_wave_epoch = 0;

    ChunkBase(Loop& loop, std::string_view family, op::chunk::Kind kind,
              KernelSource source, Binder bind, op::chunk::ArgGroups groups,
              int chunk_size, int block_size)
        : StrategyBase{loop, family, source, bind, block_size}, m_kind{kind},
          m_groups{groups}, m_chunk_size{chunk_size} {}

    Entry& entry_for(const op::loop::Configuration& configuration);
    void report(const Entry& entry) const;

public:
    op::loop::Prepared prepare(const op::loop::Configuration& configuration,
                               bool now) override;
    std::optional<float> run(const op::loop::Prepared& prepared,
                             const op::loop::Launch& launch) override;
    std::string statistics(const op::loop::Prepared& prepared) const override;
    void release() override;

    // Configurations partitioned so far, accepted or rejected.
    std::size_t entries() const { return m_entries.size(); }
};

// Increments staged in shared memory and flushed with global atomics.
class ChunkAtomics : public ChunkBase {
public:
    ChunkAtomics(Loop& loop, KernelSource source, Binder bind,
                 op::chunk::ArgGroups groups, int chunk_size, int block_size)
        : ChunkBase{loop, "chunk_atomics", op::chunk::Kind::staged, source,
                    bind, groups, chunk_size, block_size} {}
};

// Coloured chunks, then coloured elements within each, so every slot is
// seeded and stored with plain loads and stores.
class ChunkColouring : public ChunkBase {
public:
    ChunkColouring(Loop& loop, KernelSource source, Binder bind,
                   op::chunk::ArgGroups groups, int chunk_size, int block_size)
        : ChunkBase{loop, "chunk_colouring", op::chunk::Kind::coloured, source,
                    bind, groups, chunk_size, block_size} {}
};

} // namespace op::gpu
