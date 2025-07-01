#define OP_MPI_CORE_NOMPI

#include <op_mpi_cuda_unified_kernels.h>

#include <op_lib_mpi.h>
#include <op_cuda_rt_support.h>
#include <op_gpu_shims.h>

#include <cstdio>
#include <cstdint>

namespace op::mpi::unified {

constexpr int BLOCK_SIZE = 128;

void *gather_buf = nullptr;
size_t gather_buf_size = 0;

void *scatter_buf = nullptr;
size_t scatter_buf_size = 0;

gpuEvent_t gather_event;
bool gather_event_initialised = false;

static void ensure_capacity(void **buffer, size_t *size, size_t capacity) {
    if (capacity <= *size) {
        return;
    }

    if (*buffer != nullptr) {
        cutilSafeCall(gpuFreeAsync(*buffer, 0));
    }

    size_t new_size = capacity * 1.2;
    cutilSafeCall(gpuMallocAsync(buffer, new_size, 0));
    *size = new_size;
}

std::tuple<void *, void *> alloc_exchange_buffers(size_t gather_size, size_t scatter_size) {
    ensure_capacity(&gather_buf, &gather_buf_size, gather_size);
    ensure_capacity(&scatter_buf, &scatter_buf_size, scatter_size);

    return {gather_buf, scatter_buf};
}

template<typename T>
__global__ void gather_kernel(GatherSpec gather) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= gather.size) {
        return;
    }

    auto set_elem = gather.list[thread_id];
    for (int i = 0; i < gather.dat.dim; ++i) {
        ((T *) gather.target)[thread_id * gather.dat.dim + i] = gather.dat.get<T>(set_elem, i);
    }
}

void start_gather(const GatherSpec &gather) {
    if (gather.size == 0) return;
    size_t num_blocks = (gather.size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;

    switch (gather.dat.elem_size) {
        case 4: gather_kernel<std::uint32_t><<<num_blocks, BLOCK_SIZE>>>(gather); break;
        case 8: gather_kernel<std::uint64_t><<<num_blocks, BLOCK_SIZE>>>(gather); break;

        default:
            std::printf("Error: unexpected gather elem size: %d\n", gather.dat.elem_size);
            exit(1);
    }
}

template<typename T>
__global__ void scatter_kernel(ScatterSpec scatter) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= scatter.size) {
        return;
    }

    auto set_elem = scatter.is_indirect() ? scatter.list[thread_id] : scatter.offset + thread_id;
    for (int i = 0; i < scatter.dat.dim; ++i) {
        scatter.dat.get<T>(set_elem, i) = ((T *) scatter.source)[thread_id * scatter.dat.dim + i];
    }
}

void start_scatter(const ScatterSpec &scatter) {
    if (scatter.size == 0) return;
    size_t num_blocks = (scatter.size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;

    switch (scatter.dat.elem_size) {
        case 4: scatter_kernel<std::uint32_t><<<num_blocks, BLOCK_SIZE>>>(scatter); break;
        case 8: scatter_kernel<std::uint64_t><<<num_blocks, BLOCK_SIZE>>>(scatter); break;

        default:
            std::printf("Error: unexpected scatter elem size: %d\n", scatter.dat.elem_size);
            exit(1);
    }
}

void record_gathers() {
    if (!gather_event_initialised) {
        cutilSafeCall(gpuEventCreateWithFlags(&gather_event, gpuEventDisableTiming));
    }

    cutilSafeCall(gpuEventRecord(gather_event, 0));
}

void wait_gathers() {
    cutilSafeCall(gpuEventSynchronize(gather_event));
}

}
