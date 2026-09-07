#define OP_MPI_CORE_NOMPI

#include <op_mpi_unified_backend.h>

#include <op_lib_mpi.h>
#include <op_cuda_rt_support.h>
#include <op_gpu_shims.h>

#include <array>
#include <cstdio>
#include <cstdint>

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;

#define __grid_constant__
#endif

namespace op::unified_exchanges {

constexpr int BLOCK_SIZE = 128;

// Up to this many specs are passed straight to the kernel as a by-value array;
// beyond it they are staged through device memory. The cap is a portability
// limit, not a tuning knob: 64 specs is 3332 bytes of kernel parameters, and
// CUDA allowed only 4 KB before 12.1 (HIP's kernarg segment is similarly small,
// and __grid_constant__ is compiled away there). A 128 rung would need 6660
// bytes. Raising it buys almost nothing anyway - parameter size barely moves the
// launch cost, measured at 3.1 us for 416 bytes against 3.7 us for 6.6 KB.
constexpr size_t max_inline_specs = 64;

class CudaBackend final : public Backend {
public:
    // The gather and scatter kernels are asynchronous.
    CudaBackend() : Backend{false} {}

    DatPlacement placement(op_dat dat, op_map partial_map) const override;

    ExchangeBuffers alloc_buffers(size_t gather_size, size_t scatter_size) override;

    void initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) override;
    void initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) override;

    void wait_gathers() override;
    void wait_scatters() override;

private:
    // -1 until the first exchange, then 0 or 1. When 0 the exchange is staged
    // through pinned host buffers, because MPI cannot read device memory.
    int gpu_direct = -1;

    void *gather_buf = nullptr;
    size_t gather_buf_size = 0;

    void *scatter_buf = nullptr;
    size_t scatter_buf_size = 0;

    void *gather_host = nullptr;
    size_t gather_host_size = 0;

    void *scatter_host = nullptr;
    size_t scatter_host_size = 0;

    size_t gather_used = 0;
    size_t scatter_used = 0;

    gpuEvent_t gather_event;
    bool gather_event_initialised = false;

    gpuEvent_t scatter_event;
    bool scatter_event_initialised = false;

    GatherSpec *gathers_d = nullptr;
    size_t gathers_size = 0;

    int *gather_disps_d = nullptr;
    size_t gather_disps_size = 0;

    ScatterSpec *scatters_d = nullptr;
    size_t scatters_size = 0;

    int *scatter_disps_d = nullptr;
    size_t scatter_disps_size = 0;
};

// The soa flag depends only on the dat's type string and dim, so compute the
// strstr once per dat rather than on every exchange.
static std::vector<signed char> soa_cache;

static bool dat_is_soa(op_dat dat) {
    if ((std::size_t) dat->index >= soa_cache.size()) soa_cache.resize(dat->index + 1, -1);
    if (soa_cache[dat->index] < 0)
        soa_cache[dat->index] =
            (strstr(dat->type, ":soa") != NULL || (OP_auto_soa && dat->dim > 1)) ? 1 : 0;

    return soa_cache[dat->index] != 0;
}

DatPlacement CudaBackend::placement(op_dat dat, op_map partial_map) const {
    int stride = round32(dat->set->size + OP_import_exec_list[dat->set->index]->size
                                        + OP_import_nonexec_list[dat->set->index]->size);

    DatPlacement placement;
    placement.dat = DatAccessor((void *) dat->data_d, dat->dim, stride, dat->size / dat->dim,
                                dat_is_soa(dat));

    if (partial_map != nullptr) {
        placement.nonexec_export_permap = export_nonexec_list_partial_d[partial_map->index];
        placement.nonexec_import_permap = import_nonexec_list_partial_d[partial_map->index];
    } else {
        placement.exec_export = export_exec_list_d[dat->set->index];
        placement.nonexec_export = export_nonexec_list_d[dat->set->index];
    }

    return placement;
}

Backend *device_backend() {
    static CudaBackend backend;
    return &backend;
}

static void ensure_capacity(void **buffer, size_t *size, size_t capacity, bool async = true) {
    if (capacity <= *size) {
        return;
    }

    if (*buffer != nullptr) {
        if (async) {
            cutilSafeCall(gpuFreeAsync(*buffer, 0));
        } else {
            cutilSafeCall(gpuFree(*buffer));
        }
    }

    size_t new_size = capacity * 1.2;

    if (async) {
        cutilSafeCall(gpuMallocAsync(buffer, new_size, 0));
    } else {
        cutilSafeCall(gpuMalloc(buffer, new_size));
    }

    *size = new_size;
}

static void ensure_host_capacity(void **buffer, size_t *size, size_t capacity) {
    if (capacity <= *size) {
        return;
    }

    if (*buffer != nullptr) {
        cutilSafeCall(gpuHostFree(*buffer));
    }

    size_t new_size = capacity * 1.2;
    cutilSafeCall(gpuHostMalloc(buffer, new_size));

    *size = new_size;
}

ExchangeBuffers CudaBackend::alloc_buffers(size_t gather_size, size_t scatter_size) {
    if (gpu_direct < 0) {
        gpu_direct = mpi_supports_device_buffers() ? 1 : 0;
    }

    ensure_capacity(&gather_buf, &gather_buf_size, gather_size, false);
    ensure_capacity(&scatter_buf, &scatter_buf_size, scatter_size, false);

    gather_used = gather_size;
    scatter_used = scatter_size;

    if (gpu_direct) {
        return {gather_buf, scatter_buf, gather_buf, scatter_buf};
    }

    ensure_host_capacity(&gather_host, &gather_host_size, gather_size);
    ensure_host_capacity(&scatter_host, &scatter_host_size, scatter_size);

    return {gather_buf, scatter_buf, gather_host, scatter_host};
}

template<typename GathersT, typename DispsT>
__global__ void gather_kernel(__grid_constant__ const GathersT gathers,
                              __grid_constant__ const DispsT disps,
                              __grid_constant__ const int num_gathers) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int lb = cub::LowerBound(disps, num_gathers, thread_id + 1);
    if (lb == num_gathers) return;

    auto& gather = gathers[lb];
    auto index = lb > 0 ? thread_id - disps[lb - 1] : thread_id;

    gather_element(gather, index);
}

template<unsigned N>
void initiate_gathers_array(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) {
    std::array<GatherSpec, N> gathers;
    std::array<int, N> disps;

    int num_gathers = 0;
    size_t total_gather_size = 0;
    for (auto &[neighbour, gathers_batch] : gathers_for_neighbour) {
        for (auto& gather : gathers_batch) {
            gathers[num_gathers] = gather;

            total_gather_size += gather.size;
            disps[num_gathers] = total_gather_size;

            ++num_gathers;
        }
    }

    size_t num_blocks = (total_gather_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    if (num_blocks == 0) return;

    gather_kernel<<<num_blocks, BLOCK_SIZE>>>(gathers, disps, num_gathers);
}

void CudaBackend::initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) {
    if (gathers_for_neighbour.size() == 0) return;

    size_t num_gathers = 0;
    for (auto &[neighbour, gathers_batch] : gathers_for_neighbour) {
        num_gathers += gathers_batch.size();
    }

    if (num_gathers <= max_inline_specs) {
        if      (num_gathers <= 4)  initiate_gathers_array<4>(gathers_for_neighbour);
        else if (num_gathers <= 8)  initiate_gathers_array<8>(gathers_for_neighbour);
        else if (num_gathers <= 16) initiate_gathers_array<16>(gathers_for_neighbour);
        else if (num_gathers <= 32) initiate_gathers_array<32>(gathers_for_neighbour);
        else                        initiate_gathers_array<64>(gathers_for_neighbour);
    } else {
        size_t total_gather_size = 0;
        std::vector<GatherSpec> gathers;
        std::vector<int> disps;

        for (auto &[neighbour, gathers_batch] : gathers_for_neighbour) {
            for (auto& gather : gathers_batch) {
                gathers.push_back(gather);
                total_gather_size += gather.size;
                disps.push_back(total_gather_size);
            }
        }

        ensure_capacity((void **) &gathers_d, &gathers_size, sizeof(GatherSpec) * gathers.size());
        cutilSafeCall(gpuMemcpyAsync((void *) gathers_d, (void *) gathers.data(),
                                     sizeof(GatherSpec) * gathers.size(), gpuMemcpyHostToDevice));


        ensure_capacity((void **) &gather_disps_d, &gather_disps_size, sizeof(int) * disps.size());
        cutilSafeCall(gpuMemcpyAsync((void *) gather_disps_d, (void *) disps.data(),
                                     sizeof(int) * disps.size(), gpuMemcpyHostToDevice));

        size_t num_blocks = (total_gather_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
        if (num_blocks > 0) {
            gather_kernel<<<num_blocks, BLOCK_SIZE>>>(gathers_d, gather_disps_d, gathers.size());
        }
    }

    if (!gpu_direct && gather_used > 0) {
        cutilSafeCall(gpuMemcpyAsync(gather_host, gather_buf, gather_used,
                                     gpuMemcpyDeviceToHost, 0));
    }

    if (!gather_event_initialised) {
        cutilSafeCall(gpuEventCreateWithFlags(&gather_event, gpuEventDisableTiming));
        gather_event_initialised = true;
    }

    cutilSafeCall(gpuEventRecord(gather_event, 0));
}

void CudaBackend::wait_gathers() {
    cutilSafeCall(gpuEventSynchronize(gather_event));
}

template<typename ScattersT, typename DispsT>
__global__ void scatter_kernel(__grid_constant__ const ScattersT scatters,
                               __grid_constant__ const DispsT disps,
                               __grid_constant__ const int num_scatters) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int lb = cub::LowerBound(disps, num_scatters, thread_id + 1);
    if (lb == num_scatters) return;

    auto& scatter = scatters[lb];
    auto index = lb > 0 ? thread_id - disps[lb - 1] : thread_id;

    scatter_element(scatter, index);
}

template<unsigned N>
void initiate_scatters_array(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) {
    std::array<ScatterSpec, N> scatters;
    std::array<int, N> disps;

    int num_scatters = 0;
    size_t total_scatter_size = 0;
    for (auto &[neighbour, scatters_batch] : scatters_for_neighbour) {
        for (auto& scatter : scatters_batch) {
            scatters[num_scatters] = scatter;

            total_scatter_size += scatter.size;
            disps[num_scatters] = total_scatter_size;

            ++num_scatters;
        }
    }

    size_t num_blocks = (total_scatter_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    if (num_blocks == 0) return;

    scatter_kernel<<<num_blocks, BLOCK_SIZE>>>(scatters, disps, num_scatters);
}

void CudaBackend::initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) {
    if (scatters_for_neighbour.size() == 0) return;

    size_t num_scatters = 0;
    for (auto &[neighbour, scatters_batch] : scatters_for_neighbour) {
        num_scatters += scatters_batch.size();
    }

    if (!gpu_direct && scatter_used > 0) {
        cutilSafeCall(gpuMemcpyAsync(scatter_buf, scatter_host, scatter_used,
                                     gpuMemcpyHostToDevice, 0));
    }

    if (num_scatters <= max_inline_specs) {
        if      (num_scatters <= 4)  initiate_scatters_array<4>(scatters_for_neighbour);
        else if (num_scatters <= 8)  initiate_scatters_array<8>(scatters_for_neighbour);
        else if (num_scatters <= 16) initiate_scatters_array<16>(scatters_for_neighbour);
        else if (num_scatters <= 32) initiate_scatters_array<32>(scatters_for_neighbour);
        else                         initiate_scatters_array<64>(scatters_for_neighbour);
    } else {
        size_t total_scatter_size = 0;
        std::vector<ScatterSpec> scatters;
        std::vector<int> disps;

        for (auto &[neighbour, scatters_batch] : scatters_for_neighbour) {
            for (auto& scatter : scatters_batch) {
                scatters.push_back(scatter);
                total_scatter_size += scatter.size;
                disps.push_back(total_scatter_size);
            }
        }

        ensure_capacity((void **) &scatters_d, &scatters_size, sizeof(ScatterSpec) * scatters.size());
        cutilSafeCall(gpuMemcpyAsync((void *) scatters_d, (void *) scatters.data(),
                                     sizeof(ScatterSpec) * scatters.size(), gpuMemcpyHostToDevice));

        ensure_capacity((void **) &scatter_disps_d, &scatter_disps_size, sizeof(int) * disps.size());
        cutilSafeCall(gpuMemcpyAsync((void *) scatter_disps_d, (void *) disps.data(),
                                     sizeof(int) * disps.size(), gpuMemcpyHostToDevice));


        size_t num_blocks = (total_scatter_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
        if (num_blocks > 0) {
            scatter_kernel<<<num_blocks, BLOCK_SIZE>>>(scatters_d, scatter_disps_d, scatters.size());
        }
    }

    if (!scatter_event_initialised) {
        cutilSafeCall(gpuEventCreateWithFlags(&scatter_event, gpuEventDisableTiming));
        scatter_event_initialised = true;
    }

    cutilSafeCall(gpuEventRecord(scatter_event, 0));
}

void CudaBackend::wait_scatters() {
    if (!scatter_event_initialised) {
        return;
    }

    cutilSafeCall(gpuEventSynchronize(scatter_event));
}

}
