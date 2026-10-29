#pragma once

// The OpenMP backend's loop, built from the generated code's descriptor
// (op2/rt/abi/cpu/descriptor.h) on its first call: the base loop, plus the
// call's sections, threads, globals and chunk scratch.  Defined in
// src/rt/cpu/op_cpu_loop.cpp.

#include <op2/rt/abi/cpu/descriptor.h>
#include <op2/rt/abi/cpu/launch.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/loop.h>
#include <op_lib_core.h>

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace op::rt::cpu {

class Globals;

class Loop : public op::rt::loop::Loop {
private:
    int (*m_max_threads)();
    // The call under way: its sections, its threads, their globals and
    // chunk scratch.
    op::rt::loop::Sections m_sections;
    int m_threads = 1;
    std::unique_ptr<Globals> m_globals;
    std::vector<std::unique_ptr<char[]>> m_scratch;
    std::vector<char *> m_scratch_lines;
    std::size_t m_scratch_capacity = 0;

public:
    explicit Loop(const abi::cpu::Loop& descriptor);
    ~Loop() override;

    // --- For the strategies ---

    const op::rt::loop::Sections& call_sections() const { return m_sections; }

    // A range and a chunk phase over the call's threads and their globals,
    // the chunk phase's with their scratch.
    abi::cpu::Range range() const;
    abi::cpu::Chunk chunks() const;

    // Give every thread bytes of chunk scratch, aligned to a cache line.
    void reserve_scratch(std::size_t bytes);


protected:
    // --- Per call ---

    // The call's sections, and each thread's copy of the globals.
    void begin_call(const op::rt::loop::Call& call, const op::rt::tune::Autotuner::Decision&,
                    op_arg *args, int nargs) override;

    // Combine the threads' copies into the arguments.
    void combine_globals(op_arg *args, int nargs) override;
};

} // namespace op::rt::cpu
