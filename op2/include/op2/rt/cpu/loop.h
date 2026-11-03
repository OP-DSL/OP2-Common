#pragma once

// The OpenMP backend's loop, built from the generated code's descriptor
// (op2/rt/abi/cpu/descriptor.h) on its first call: the base loop, plus the
// call's threads, globals and chunk scratch.  Defined in
// src/rt/cpu/op_cpu_loop.cpp.

#include <op2/rt/abi/cpu/descriptor.h>
#include <op2/rt/abi/cpu/launch.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/loop.h>
#include <op_lib_core.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
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
    std::unique_ptr<char[]> m_storage;
    std::size_t m_capacity = 0;
    std::vector<char *> m_pointers;

    template<typename T>
    void combine(std::span<op_arg> args, int index) const;

public:
    // Lay out and initialise every thread's copies: increments from zero,
    // the rest from the argument's value.
    void prepare(std::span<op_arg> args, int threads);

    // Per thread its copy of each argument, nargs apart; null for an
    // argument not kept per thread.
    char *const *pointers() const { return m_pointers.data(); }
    int nargs() const { return m_nargs; }

    // Combine every thread's reductions into the arguments, in thread order.
    void combine(std::span<op_arg> args, const char *loop) const;
};

class Loop : public op::rt::loop::Loop {
private:
    int (*m_max_threads)();
    // The call under way: its threads, their globals and chunk scratch.
    int m_threads = 1;
    Globals m_globals;
    std::vector<std::unique_ptr<char[]>> m_scratch;
    std::vector<char *> m_scratch_lines;
    std::size_t m_scratch_capacity = 0;

public:
    explicit Loop(const abi::cpu::Loop& descriptor);
    ~Loop() override;

    // --- For the strategies ---

    // A range and a chunk phase over the call's threads and their globals,
    // the chunk phase's with their scratch.
    abi::cpu::Range range() const;
    abi::cpu::Chunk chunks() const;

    // Give every thread bytes of chunk scratch, aligned to a cache line.
    void reserve_scratch(std::size_t bytes);

    // Combine the threads' copies into the arguments.
    void combine_globals(const op::rt::loop::Call& call);

protected:
    // --- Per call ---

    // The call's threads, and each one's copy of the globals.
    void begin_call(const op::rt::loop::Call& call) override;
};

} // namespace op::rt::cpu
