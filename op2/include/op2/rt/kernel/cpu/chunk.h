#pragma once

// The chunked strategies' OpenMP region: each batch's chunks shared out over
// the threads (op2/rt/abi/cpu/launch.h), a batch once the one before is done,
// under chunk_colouring through chunk_order.
// make(thread) gives a thread its chunk function, chunk(chunk, begin, end),
// once per region; under chunk_atomics it stages the chunk's targets in the
// thread's scratch and flushes them, under chunk_colouring it applies the
// chunk's elements in place.

#include <op2/rt/abi/cpu/launch.h>

#include <omp.h>

namespace op::rt::kernel::cpu {

template<bool ordered, typename Make>
void chunks(const op::rt::abi::cpu::Chunk& record, Make&& make) {
#pragma omp parallel num_threads(record.threads)
    {
        const int thread = omp_get_thread_num();
        auto chunk = make(op::rt::abi::cpu::Thread{
            record.globals + thread * record.nargs,
            record.scratch != nullptr ? record.scratch[thread] : nullptr});
        // The calling thread tests the halo exchange between its chunks.
        const bool tests = thread == 0 && record.progress.test != nullptr;
        int untested = 0;
        for (int batch = 0; batch < record.batches; ++batch) {
            const int first = record.batch_offsets[batch];
            const int last = record.batch_offsets[batch + 1];
#pragma omp for schedule(static)
            for (int position = first; position < last; ++position) {
                const int index = ordered ? record.chunk_order[position] : position;
                const int begin = record.chunk_offsets[index];
                const int end = record.chunk_offsets[index + 1];
                chunk(index, begin, end);
                if (tests && (untested += end - begin) >= record.progress.every) {
                    record.progress.test();
                    untested = 0;
                }
            }
        }
    }
}

} // namespace op::rt::kernel::cpu
