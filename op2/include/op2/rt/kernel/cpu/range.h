#pragma once

// The plain, atomics and colouring strategies' OpenMP region: a range of
// elements (op2/rt/abi/cpu/launch.h) in a contiguous share per thread.
// make(thread) gives a thread its element function, element(n), once per
// region, so the thread's own setup is done once.

#include <op2/rt/abi/cpu/launch.h>

#include <omp.h>

namespace op::rt::kernel::cpu {

template<bool ordered, typename Make>
void range(const op::rt::abi::cpu::Range& record, Make&& make) {
    const long long size = static_cast<long long>(record.end) - record.begin;
#pragma omp parallel num_threads(record.threads)
    {
        const long long thread = omp_get_thread_num();
        const long long count = omp_get_num_threads();
        auto element = make(op::rt::abi::cpu::Thread{record.globals + thread * record.nargs});
        const int begin = static_cast<int>(record.begin + size * thread / count);
        const int end = static_cast<int>(record.begin + size * (thread + 1) / count);
        // The calling thread tests the halo exchange between steps.
        const bool tests = thread == 0 && record.progress.test != nullptr;
        const int step = tests ? record.progress.every : end - begin;
        for (int first = begin; first < end;) {
            const int last = end - first > step ? first + step : end;
            for (int i = first; i < last; ++i)
                element(ordered ? record.order[i] : i);
            if (tests)
                record.progress.test();
            first = last;
        }
    }
}

} // namespace op::rt::kernel::cpu
