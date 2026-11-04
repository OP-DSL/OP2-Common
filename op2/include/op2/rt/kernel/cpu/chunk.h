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
        for (int batch = 0; batch < record.batches; ++batch) {
            const int first = record.batch_offsets[batch];
            const int last = record.batch_offsets[batch + 1];
#pragma omp for schedule(static)
            for (int position = first; position < last; ++position) {
                const int index = ordered ? record.chunk_order[position] : position;
                chunk(index, record.chunk_offsets[index], record.chunk_offsets[index + 1]);
            }
        }
    }
}

} // namespace op::rt::kernel::cpu
