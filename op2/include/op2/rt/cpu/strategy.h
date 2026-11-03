#pragma once

// The OpenMP strategies: each schedules a loop's elements over host threads
// in phases, each a record its family's run is given, which runs it in the
// generated code's OpenMP region (op2/rt/kernel/cpu/range.h,
// op2/rt/kernel/cpu/chunk.h), compiled with the loop's kernel so the element
// loop inlines.  Defined in src/rt/cpu/op_cpu_strategy.cpp.

#include <op2/rt/cpu/chunk_plan.h>
#include <op2/rt/cpu/executor.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/plan/cache.h>
#include <op2/rt/plan/colouring.h>
#include <op_lib_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::cpu {

// --- The base of every OpenMP strategy ----------------------------------------------

class StrategyBase : public op::rt::loop::Strategy {
protected:
    Executor& m_executor;
    std::string m_name;
    const abi::cpu::Family& m_family;

    StrategyBase(Executor& executor, const op::rt::loop::Point& point,
                 const abi::cpu::Family& family)
        : m_executor{executor}, m_name{point.name}, m_family{family} {}

    // Run [begin, end), or the positions of order, over the threads.
    void split(int begin, int end, const int *order, std::span<op_arg> args);

    // Run the phases of a call through the loop's, phase(index) running one
    // phase over the threads; the phases' own time when timed.
    template<typename F>
    std::optional<float> run_phases(const op::rt::loop::Call& call, bool timed,
                                    const op::rt::loop::Phases& phases, F&& phase);

public:
    static std::vector<op::rt::loop::Axis> axes() { return {}; }

    std::string_view name() const override { return m_name; }

    // The call's threads and their copies of the globals.
    void begin_call(const op::rt::loop::Prepared&, const op::rt::loop::Call& call) override {
        m_executor.begin_call(call);
    }
};

// --- Plain and atomics ------------------------------------------------------------

// Each section split over the threads; the body names the increments: none
// for plain, atomics for atomics.
class Simple : public StrategyBase {
public:
    Simple(Executor& executor, const op::rt::loop::Point& point, const abi::cpu::Family& family)
        : StrategyBase{executor, point, family} {}

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                                   op::rt::loop::Urgency urgency,
                                   op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Call& call,
                             bool timed) override;
};

// --- Colouring -----------------------------------------------------------------------

// Each colour of a set colouring split over the threads.
class Colouring : public StrategyBase {
private:
    op::rt::plan::PlanSlots<op::rt::plan::SetColouring> m_plans;

public:
    Colouring(Executor& executor, const op::rt::loop::Point& point, const abi::cpu::Family& family)
        : StrategyBase{executor, point, family} {}

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                                   op::rt::loop::Urgency urgency,
                                   op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Call& call,
                             bool timed) override;
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
    op::rt::abi::ArgGroups m_groups;
    int m_chunk_size;
    op::rt::plan::PlanSlots<ChunkResult> m_plans;

    ChunkBase(Executor& executor, const op::rt::loop::Point& point, const abi::cpu::Family& family)
        : StrategyBase{executor, point, family}, m_groups{family.groups},
          m_chunk_size{point.value("chunk_size")} {}

    // The variant's plan of the call, and its inputs to the plan's key beyond
    // the shared ones.
    virtual ChunkResult make_plan(const op::rt::loop::Call& call) const = 0;
    virtual void add_key(op::rt::plan::PlanKey& key) const = 0;

    const ChunkResult& entry_for(const op::rt::loop::Call& call);

public:
    // Elements per chunk: OP_<FAMILY>_CHUNK_SIZES, else OP_CHUNK_SIZES.
    static std::vector<op::rt::loop::Axis> axes() {
        return {{"CHUNK_SIZES", "chunk_size", false, {default_chunk_size}}};
    }

    op::rt::loop::Prepared prepare(const op::rt::loop::Call& call,
                                   op::rt::loop::Urgency urgency,
                                   op::rt::loop::KernelKind kind) override;
    std::optional<float> run(const op::rt::loop::Prepared& prepared,
                             const op::rt::loop::Call& call,
                             bool timed) override;
    std::string statistics(const op::rt::loop::Prepared& prepared) const override;
    void forget(std::uint64_t key) override { m_plans.forget(key); }
};

class ChunkAtomics : public ChunkBase {
protected:
    ChunkResult make_plan(const op::rt::loop::Call& call) const override;
    void add_key(op::rt::plan::PlanKey& key) const override;

public:
    ChunkAtomics(Executor& executor, const op::rt::loop::Point& point,
                 const abi::cpu::Family& family)
        : ChunkBase{executor, point, family} {}
};

// Conflicts come from the call's updated dats, so it needs no groups.
class ChunkColouring : public ChunkBase {
protected:
    ChunkResult make_plan(const op::rt::loop::Call& call) const override;
    void add_key(op::rt::plan::PlanKey&) const override {}

public:
    ChunkColouring(Executor& executor, const op::rt::loop::Point& point,
                   const abi::cpu::Family& family)
        : ChunkBase{executor, point, family} {}
};

// --- Families -------------------------------------------------------------------

// A strategy family as the OpenMP backend offers it.
using OfferedFamily = op::rt::loop::OfferedFamily<Executor, abi::cpu::Family>;

// The families in ladder order: plain, chunk_atomics, atomics,
// chunk_colouring and colouring, the atomic ones off by default (an atomic
// increment is a compare-and-swap loop on most CPUs), so chunk colouring runs
// over the set colouring floor.
std::span<const OfferedFamily> strategy_families();

// Declare the families the generated code registers, in ladder order; the
// CPU libraries' op_init calls it.
void init();

} // namespace op::rt::cpu
