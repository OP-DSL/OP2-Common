#pragma once

// What the OpenMP backend's phases are given: a phase's record, as a
// strategy hands it to the generated code's OpenMP region
// (op2/rt/kernel/cpu/range.h, op2/rt/kernel/cpu/chunk.h), and a thread's own
// part of it.

#include <op2/rt/abi/staging.h>

namespace op::rt::abi::cpu {

// A chunk_atomics plan's staging, as a staged chunk reads it.
struct StagingArrays {
    // Per reference and element, stage_words[reference * stride + element].
    int stride = 0;
    const StageWord *stage_words = nullptr;
    // Per region its first slot in slot_targets, then the slot count.
    const int *slot_offsets = nullptr;
    const int *slot_targets = nullptr;
};

// A thread's own: its copy of each argument kept per thread, by argument
// index, null for the others; and its chunk scratch.
struct Thread {
    char *const *globals = nullptr;
    char *scratch = nullptr;

    template<typename T>
    T *global(int arg) const {
        return reinterpret_cast<T *>(globals[arg]);
    }
};

// A range of elements split over the threads: [begin, end), or under a set
// colouring the positions of order.  Per thread its copy of each argument
// kept per thread, nargs apart.
struct Range {
    int threads = 1;
    char *const *globals = nullptr;
    int nargs = 0;
    int begin = 0;
    int end = 0;
    const int *order = nullptr;
};

// Batches of chunks shared out over the threads, one after another: per
// batch its first position, then the last batch's end; under chunk_colouring
// per position its chunk; per chunk its first element, then the last chunk's
// end.  Per thread its globals and scratch; chunk_atomics' staging.
struct Chunk {
    int threads = 1;
    char *const *globals = nullptr;
    int nargs = 0;
    char *const *scratch = nullptr;
    int batches = 0;
    const int *batch_offsets = nullptr;
    const int *chunk_order = nullptr;
    const int *chunk_offsets = nullptr;
    StagingArrays staging;
};

} // namespace op::rt::abi::cpu
