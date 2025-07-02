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

GatherSpec *gathers_d;
size_t gathers_size = 0;

ScatterSpec *scatters_d;
size_t scatters_size = 0;

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

__global__ void gather_kernel(GatherSpec *gathers, int num_gathers) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int gather_index = 0;
    int accumulated_gather_sizes = 0;

    while (gather_index < num_gathers) {
        if (thread_id < accumulated_gather_sizes + gathers[gather_index].size) break;

        accumulated_gather_sizes += gathers[gather_index].size;
        ++gather_index;
    }

    if (gather_index == num_gathers) return;

    auto& gather = gathers[gather_index];
    auto index = thread_id - accumulated_gather_sizes;

    auto set_elem = gather.list[index];
    if (gather.dat.elem_size == 4) {
        for (int i = 0; i < gather.dat.dim; ++i) {
            ((std::uint32_t *) gather.target)[index * gather.dat.dim + i] =
                gather.dat.get<std::uint32_t>(set_elem, i);
        }
    } else {
        for (int i = 0; i < gather.dat.dim; ++i) {
            ((std::uint64_t *) gather.target)[index * gather.dat.dim + i] =
                gather.dat.get<std::uint64_t>(set_elem, i);
        }
    }
}

void initiate_gathers(const std::map<int, std::vector<GatherSpec>> &gathers_for_neighbour) {
    if (gathers_for_neighbour.size() == 0) return;

    size_t total_gather_size = 0;
    std::vector<GatherSpec> gathers;

    for (auto &[neighbour, gathers_batch] : gathers_for_neighbour) {
        for (auto& gather : gathers_batch) {
            gathers.push_back(gather);
            total_gather_size += gather.size;
        }
    }

    ensure_capacity((void **) &gathers_d, &gathers_size, sizeof(GatherSpec) * gathers.size());
    cutilSafeCall(gpuMemcpyAsync((void *) gathers_d, (void *) gathers.data(),
                                 sizeof(GatherSpec) * gathers.size(), gpuMemcpyHostToDevice));

    size_t num_blocks = (total_gather_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    gather_kernel<<<num_blocks, BLOCK_SIZE>>>(gathers_d, gathers.size());

    if (!gather_event_initialised) {
        cutilSafeCall(gpuEventCreateWithFlags(&gather_event, gpuEventDisableTiming));
    }

    cutilSafeCall(gpuEventRecord(gather_event, 0));
}

void wait_gathers() {
    cutilSafeCall(gpuEventSynchronize(gather_event));
}

__global__ void scatter_kernel(ScatterSpec *scatters, int num_scatters) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int scatter_index = 0;
    int accumulated_scatter_sizes = 0;

    while (scatter_index < num_scatters) {
        if (thread_id < accumulated_scatter_sizes + scatters[scatter_index].size) break;

        accumulated_scatter_sizes += scatters[scatter_index].size;
        ++scatter_index;
    }

    if (scatter_index == num_scatters) return;

    auto& scatter = scatters[scatter_index];
    auto index = thread_id - accumulated_scatter_sizes;

    auto set_elem = scatter.is_indirect() ? scatter.list[index] : scatter.offset + index;
    if (scatter.dat.elem_size == 4) {
        for (int i = 0; i < scatter.dat.dim; ++i) {
            scatter.dat.get<std::uint32_t>(set_elem, i) =
                ((std::uint32_t *) scatter.source)[index * scatter.dat.dim + i];
        }
    } else {
        for (int i = 0; i < scatter.dat.dim; ++i) {
            scatter.dat.get<std::uint64_t>(set_elem, i) =
                ((std::uint64_t *) scatter.source)[index * scatter.dat.dim + i];
        }
    }
}

void initiate_scatters(const std::map<int, std::vector<ScatterSpec>> &scatters_for_neighbour) {
    if (scatters_for_neighbour.size() == 0) return;

    size_t total_scatter_size = 0;
    std::vector<ScatterSpec> scatters;

    for (auto &[neighbour, scatters_batch] : scatters_for_neighbour) {
        for (auto& scatter : scatters_batch) {
            scatters.push_back(scatter);
            total_scatter_size += scatter.size;
        }
    }

    ensure_capacity((void **) &scatters_d, &scatters_size, sizeof(ScatterSpec) * scatters.size());
    cutilSafeCall(gpuMemcpyAsync((void *) scatters_d, (void *) scatters.data(),
                                 sizeof(ScatterSpec) * scatters.size(), gpuMemcpyHostToDevice, op2_grp_secondary));

    size_t num_blocks = (total_scatter_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    scatter_kernel<<<num_blocks, BLOCK_SIZE, 0, op2_grp_secondary>>>(scatters_d, scatters.size());
}

}
