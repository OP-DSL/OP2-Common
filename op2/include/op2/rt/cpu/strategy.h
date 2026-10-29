#pragma once

// The OpenMP strategies: each schedules a loop's elements over host threads
// in phases, each a record its family's run is given, which runs it in the
// generated code's OpenMP region (op2/rt/kernel/cpu/range.h,
// op2/rt/kernel/cpu/chunk.h), compiled with the loop's kernel so the element
// loop inlines.  Defined in src/rt/cpu/op_cpu_strategy.cpp; the per-thread
// globals in src/rt/cpu/op_cpu_loop.cpp.

#include <op2/rt/cpu/chunk_plan.h>
#include <op2/rt/cpu/loop.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/plan/cache.h>
#include <op2/rt/plan/colouring.h>
#include <op_lib_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::cpu {

// --- Per-thread globals ------------------------------------------------------------

// A copy per thread of each argument a loop reduces (OP_INC, OP_MIN,
// OP_MAX), uses as scratch (OP_WORK) or carries with a reduction
// (op_arg_info), combined into the arguments once the owned elements ran.
class Globals {
private:
    int m_nargs = 0;
    int m_threads = 0;
    std::size_t m_stride = 0;
    std::vector<std::size_t> m_offsets;
    std::unique_ptr<char[]> m_storage;
    std::size_t m_capacity = 0;
    std::vector<char *> m_pointers;

    template<typename T>
    void combine(op_arg *args, int index) const;

public:
    // Lay out and initialise every thread's copies: increments from zero,
    // the rest from the argument's value.
    void prepare(op_arg *args, int nargs, int threads);

    // Per thread its copy of each argument, nargs apart; null for an
    // argument not kept per thread.
    char *const *pointers() const { return m_pointers.data(); }
    int nargs() const { return m_nargs; }

    // Combine every thread's reductions into the arguments, in thread order.
    void combine(op_arg *args, const char *loop) const;
};

// --- Plans ------------------------------------------------------------------------

// A chunk plan, or why the call does not suit it.
struct ChunkEntry {
    std::string_view rejected;
    std::unique_ptr<ChunkPlan> plan;
};

// --- The base of every OpenMP strategy ----------------------------------------------

class StrategyBase : public op::rt::loop::Strategy {
protected:
    Loop& m_loop;
    std::string m_name;
    const abi::cpu::Family& m_family;

    StrategyBase(Loop& loop, const op::rt::loop::Point& point, const abi::cpu::Family& family)
        : m_loop{loop}, m_name{point.name}, m_family{family} {}

    // Split [begin, end) over the threads, a body call each.
    void split(int begin, int end, const int *order, op_arg *args);

    // Run the phases of a call through the loop's, phase(index) running one
    // phase over the threads; the phases' own time when timed.
    template<typename F>
    std::optional<float> run_phases(const op::rt::loop::RunArgs& run, int phases,
                                    int wait_before, int globals_after, F&& phase);

public:
    static std::vector<op::rt::loop::Axis> axes() { return {}; }

    std::string_view name() const override { return m_name; }
};

// --- Plain and atomics ------------------------------------------------------------

// Each section split over the threads; the body names the increments: none
// for plain, atomics for atomics.
class Simple : public StrategyBase {
public:
    Simple(Loop& loop, const op::rt::loop::Point& point, const abi::cpu::Family& family)
        : StrategyBase{loop, point, family} {}

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                                   op::rt::loop::Urgency urgency,
                                   op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::RunArgs& run) override;
};

// --- Colouring -----------------------------------------------------------------------

// Each colour of a set colouring split over the threads.
class Colouring : public StrategyBase {
private:
    op::rt::plan::PlanSlots<op::rt::plan::SetColouring> m_plans;

public:
    Colouring(Loop& loop, const op::rt::loop::Point& point, const abi::cpu::Family& family)
        : StrategyBase{loop, point, family} {}

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                                   op::rt::loop::Urgency urgency,
                                   op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::RunArgs& run) override;
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;
    void forget(std::uint64_t key) override { m_plans.forget(key); }
};

// --- The chunked strategies -------------------------------------------------------

// A chunk plan (op2/rt/cpu/chunk_plan.h): each batch's chunks shared out over
// the threads, a body call per chunk.  chunk_colouring's batches share no
// target, so its chunks run the plain body over their elements;
// chunk_atomics' chunks accumulate into the thread's scratch and flush it
// with atomics.
class ChunkBase : public StrategyBase {
protected:
    bool m_coloured;
    op::rt::abi::ArgGroups m_groups;
    int m_chunk_size;
    op::rt::plan::PlanSlots<ChunkEntry> m_plans;

    ChunkBase(Loop& loop, const op::rt::loop::Point& point, bool coloured,
              const abi::cpu::Family& family)
        : StrategyBase{loop, point, family}, m_coloured{coloured}, m_groups{family.groups},
          m_chunk_size{point.value("chunk_size")} {}

    const ChunkEntry& entry_for(const op::rt::loop::Call& call);

public:
    // Elements per chunk: OP_<FAMILY>_CHUNK_SIZES, else OP_CHUNK_SIZES.
    static std::vector<op::rt::loop::Axis> axes() {
        return {{"CHUNK_SIZES", "chunk_size", false, {default_chunk_size}}};
    }

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                                   op::rt::loop::Urgency urgency,
                                   op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::RunArgs& run) override;
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;
    void forget(std::uint64_t key) override { m_plans.forget(key); }
};

class ChunkAtomics : public ChunkBase {
public:
    ChunkAtomics(Loop& loop, const op::rt::loop::Point& point, const abi::cpu::Family& family)
        : ChunkBase{loop, point, false, family} {}
};

// Conflicts come from the call's updated dats, so it needs no groups.
class ChunkColouring : public ChunkBase {
public:
    ChunkColouring(Loop& loop, const op::rt::loop::Point& point,
                   const abi::cpu::Family& family)
        : ChunkBase{loop, point, true, family} {}
};

// A strategy of family at point.
std::unique_ptr<op::rt::loop::Strategy> make_strategy(Loop& loop, const abi::cpu::Family& family,
                                                      const op::rt::loop::Point& point);

// Declare the families the generated code registers, in ladder order; the
// CPU libraries' op_init calls it.
void init();

} // namespace op::rt::cpu
