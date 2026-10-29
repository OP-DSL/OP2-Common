#pragma once

// What the OpenMP backend's phases are given: a phase as a strategy hands it
// to the generated code's OpenMP region (op2/rt/kernel/cpu/launch.h), and a
// thread's share of it as the generated body reads it.

#include <op2/rt/abi/staging.h>
#include <op_lib_core.h>

namespace op::rt::abi::cpu {

// A chunk_atomics plan's staging, as a staged body reads it.
struct StagingArrays {
    // Per reference and element, stage_words[reference * stride + element].
    int stride = 0;
    const StageWord *stage_words = nullptr;
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

} // namespace op::rt::abi::cpu
