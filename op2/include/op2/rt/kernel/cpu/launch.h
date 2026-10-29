#pragma once

// The OpenMP regions the strategies' phases run in, compiled into op2_kernels
// with the body each runs, which so inlines into it.  The generated code
// registers split<body> or chunks<body> as a family's run
// (op2/rt/cpu/loop.h), and max_threads with its loop.

#include <op2/rt/cpu/loop.h>
#include <op_lib_core.h>

#include <omp.h>

namespace op::rt::kernel::cpu {

// [phase.begin, phase.end) in a contiguous share per thread, a body call
// each.
template<auto body>
void split(const op::rt::cpu::Phase& phase, op_arg *args) {
    const long long size = static_cast<long long>(phase.end) - phase.begin;
#pragma omp parallel num_threads(phase.threads)
    {
        const long long thread = omp_get_thread_num();
        const long long count = omp_get_num_threads();
        op::rt::cpu::Work work;
        work.begin = static_cast<int>(phase.begin + size * thread / count);
        work.end = static_cast<int>(phase.begin + size * (thread + 1) / count);
        work.order = phase.order;
        work.globals = phase.globals + thread * phase.nargs;
        if (work.begin < work.end)
            body(work, args);
    }
}

// Each batch's chunks shared out over the threads, a body call per chunk;
// a batch starts once the one before is done.
template<auto body>
void chunks(const op::rt::cpu::Phase& phase, op_arg *args) {
#pragma omp parallel num_threads(phase.threads)
    {
        const int thread = omp_get_thread_num();
        op::rt::cpu::Work work;
        work.globals = phase.globals + thread * phase.nargs;
        work.scratch = phase.scratch != nullptr ? phase.scratch[thread] : nullptr;
        work.staging = phase.staging;
        for (int batch = 0; batch < phase.batches; ++batch) {
            const int first = phase.batch_offsets[batch];
            const int last = phase.batch_offsets[batch + 1];
#pragma omp for schedule(static)
            for (int position = first; position < last; ++position) {
                work.chunk = phase.chunk_order != nullptr ? phase.chunk_order[position]
                                                          : position;
                work.begin = phase.chunk_offsets[work.chunk];
                work.end = phase.chunk_offsets[work.chunk + 1];
                body(work, args);
            }
        }
    }
}

// The threads a phase may use.
inline int max_threads() { return omp_get_max_threads(); }

} // namespace op::rt::kernel::cpu
