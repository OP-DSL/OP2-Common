#pragma once

// The plain, atomics and colouring strategies' OpenMP region: a range of
// elements (op2/rt/abi/cpu/launch.h) in a contiguous share per thread.
// make(thread) gives a thread its element function, element(n), once per
// region, so the thread's own setup is done once.

#include <op2/rt/abi/cpu/launch.h>

#include <omp.h>

namespace op::rt::kernel::cpu {

template<bool ordered, typename Make>
void range(const op::rt::abi::cpu::Range& range, Make&& make) {
    const long long size = static_cast<long long>(range.end) - range.begin;
#pragma omp parallel num_threads(range.threads)
    {
        const long long thread = omp_get_thread_num();
        const long long count = omp_get_num_threads();
        const int begin = static_cast<int>(range.begin + size * thread / count);
        const int end = static_cast<int>(range.begin + size * (thread + 1) / count);
        if (begin < end) {
            auto element = make(op::rt::abi::cpu::Thread{range.globals + thread * range.nargs});
            for (int i = begin; i < end; ++i)
                element(ordered ? range.order[i] : i);
        }
    }
}

} // namespace op::rt::kernel::cpu
