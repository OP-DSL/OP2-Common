#pragma once

// The OpenMP backend as op2_kernels sees it: the loop the generated code
// builds per par_loop and registers its bodies on, and what a body and the
// OpenMP region running it (op2/rt/kernel/cpu/launch.h) are given.  The loop
// adds to the base loop the call's sections, threads, globals and chunk
// scratch; it and its strategies are defined in src/rt/cpu.

#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/staging.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/loop.h>
#include <op_lib_core.h>

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace op::rt::cpu {

class Globals;

// A chunk_atomics plan's staging, as a staged body reads it.
struct StagingArrays {
    // Per reference and element, stage_words[reference * stride + element].
    int stride = 0;
    const op::rt::abi::StageWord *stage_words = nullptr;
    // Per region its first slot in slot_targets, then the slot count.
    const int *slot_offsets = nullptr;
    const int *slot_targets = nullptr;
};

// One thread's share of a phase, as a generated body reads it.
struct Work {
    // The elements, or under a set colouring the positions of order.
    int begin = 0;
    int end = 0;
    const int *order = nullptr;
    // The calling thread's copy of each argument kept per thread, by
    // argument index; null for the others.
    char *const *globals = nullptr;
    // chunk_atomics: the chunk, the thread's scratch and the staging.
    int chunk = 0;
    char *scratch = nullptr;
    StagingArrays staging;

    template<typename T>
    T *global(int arg) const {
        return reinterpret_cast<T *>(globals[arg]);
    }
};

// One phase as a strategy hands it to the OpenMP region: the threads and
// their globals, and either elements to split over the threads or batches
// of chunks to share out among them.
struct Phase {
    int threads = 1;
    // Per thread its copy of each argument kept per thread, nargs apart.
    char *const *globals = nullptr;
    int nargs = 0;

    // A split: [begin, end), or under a set colouring the positions of
    // order.
    int begin = 0;
    int end = 0;
    const int *order = nullptr;

    // Batches of chunks, one after another: per batch its first position,
    // then the last batch's end; per position its chunk, the position itself
    // when null; per chunk its first element, then the last chunk's end.
    int batches = 0;
    const int *batch_offsets = nullptr;
    const int *chunk_order = nullptr;
    const int *chunk_offsets = nullptr;
    // chunk_atomics: per thread its scratch, and the staging.
    char *const *scratch = nullptr;
    StagingArrays staging;
};

// Run a phase over the threads, the generated body for each share.
using Run = void (*)(const Phase& phase, op_arg *args);

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
    // max_threads is op2_kernels' omp_get_max_threads.
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, bool direct, int (*max_threads)());
    ~Loop() override;

    // --- Registration, by the generated code ---

    // Add a family whose strategies run their phases through run;
    // chunk_atomics' groups are the arguments it stages.
    void add(std::string_view family, Run run, op::rt::abi::ArgGroups groups = {});

    // --- For the strategies ---

    const op::rt::loop::Sections& call_sections() const { return m_sections; }

    // A phase over the call's threads and their globals.
    Phase phase() const;

    // Give every thread bytes of chunk scratch, aligned to a cache line.
    void reserve_scratch(std::size_t bytes);

    // Per thread its scratch.
    char *const *scratch() const { return m_scratch_lines.data(); }

protected:
    // --- Per call ---

    // The call's sections, and each thread's copy of the globals.
    void begin_call(const op::rt::loop::Call& call, const op::rt::tune::Autotuner::Decision&,
                    op_arg *args, int nargs) override;

    // Combine the threads' copies into the arguments.
    void combine_globals(op_arg *args, int nargs) override;
};

} // namespace op::rt::cpu
