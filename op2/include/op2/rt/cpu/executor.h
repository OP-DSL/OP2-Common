#pragma once

// The OpenMP backend's executor, made with its loop from the generated
// code's descriptor (op2/rt/abi/cpu/descriptor.h): the call's threads, their
// globals and chunk scratch.  Defined in src/rt/cpu/op_cpu_executor.cpp.

#include <op2/rt/abi/cpu/descriptor.h>
#include <op2/rt/abi/cpu/launch.h>
#include <op2/rt/jit/params.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/ladder.h>
#include <op2/rt/loop/loop.h>
#include <op_lib_core.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace op::rt::cpu {

// A copy per thread of each argument a loop reduces (OP_INC, OP_MIN,
// OP_MAX), uses as scratch (OP_WORK) or carries with a reduction
// (op_arg_info), combined into the arguments once the owned elements ran.
class Globals {
private:
    int m_nargs = 0;
    int m_threads = 0;
    std::size_t m_stride = 0;
    std::vector<std::size_t> m_offsets;
    // Per argument, the element type it reduces in; none if it reduces none.
    std::vector<std::optional<abi::ElementType>> m_types;
    std::unique_ptr<char[]> m_storage;
    std::size_t m_capacity = 0;
    std::vector<char *> m_pointers;

    template<typename T>
    void combine(std::span<op_arg> args, int index) const;

public:
    // Lay out and initialise every thread's copies: increments from zero,
    // the rest from the argument's value.  A reduction of a type it cannot
    // combine stops the run, naming loop.
    void prepare(std::span<op_arg> args, int threads, const char *loop);

    // Per thread its copy of each argument, nargs apart; null for an
    // argument not kept per thread.
    char *const *pointers() const { return m_pointers.data(); }
    int nargs() const { return m_nargs; }

    // Combine every thread's reductions into the arguments, in thread order.
    void combine(std::span<op_arg> args) const;
};

class Executor final : public op::rt::loop::Executor {
private:
    const abi::cpu::Loop& m_descriptor;
    // The call under way: its threads, their globals and chunk scratch.
    int m_threads = 1;
    Globals m_globals;
    std::vector<std::unique_ptr<char[]>> m_scratch;
    std::vector<char *> m_scratch_lines;
    std::size_t m_scratch_capacity = 0;

public:
    // The executor of descriptor's loop, whose kernels take no parameters.
    Executor(const abi::cpu::Loop& descriptor, op::rt::jit::ParamSet& params);

    void add_strategies(op::rt::loop::Ladder& ladder) override;

    // --- For the strategies ---

    const char *name() const { return m_descriptor.name; }

    // Start a run of call: its threads, and each one's copy of the globals.
    void begin_call(const op::rt::loop::Call& call);

    // A range and a chunk phase over the call's threads and their globals,
    // the chunk phase's with their scratch.
    abi::cpu::Range range() const;
    abi::cpu::Chunk chunks() const;

    // Give every thread bytes of chunk scratch, aligned to a cache line.
    void reserve_scratch(std::size_t bytes);

    // Combine the threads' copies into the arguments.
    void combine_globals(const op::rt::loop::Call& call);
};

} // namespace op::rt::cpu
