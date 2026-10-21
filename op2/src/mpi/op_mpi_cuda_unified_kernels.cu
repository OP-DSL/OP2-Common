#define OP_MPI_CORE_NOMPI

#include <op_mpi_unified_backend.h>

#include <op_lib_mpi.h>
#include <op_mpi_halo.h>
#include <op_cuda_rt_support.h>
#include <gpu/shims.h>

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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

    bool asynchronous() const override { return true; }
    bool grows(int slot, size_t gather_size, size_t scatter_size) const override;
    ExchangeBuffers slot_buffers(int slot, size_t gather_size, size_t scatter_size) override;
    void slot_gathers(int slot, const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) override;
    void slot_scatters(int slot, unsigned seq,
                       const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) override;
    bool gathered(int slot) override;
    bool scattered(int slot) override;
    void raise(int slot, unsigned seq) override;

private:
    struct Slot {
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
    };

    std::array<Slot, slots> async_slots;
    // Each slot's flag, in mapped pinned memory, at its device address too.
    volatile unsigned *flags = nullptr;
    void *flags_d = nullptr;

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
    int stride = round32(dat->set->size + OP_set_halos[dat->set->index].import_exec.size()
                                        + OP_set_halos[dat->set->index].import_nonexec.size());

    DatPlacement placement;
    placement.dat = DatAccessor((void *) dat->data_d, dat->dim, stride, dat->size / dat->dim,
                                dat_is_soa(dat));

    if (partial_map != nullptr) {
        placement.nonexec_export_permap = OP_map_halos_d[partial_map->index].export_nonexec.get();
        placement.nonexec_import_permap = OP_map_halos_d[partial_map->index].import_nonexec.get();
    } else {
        placement.exec_export = OP_set_halos_d[dat->set->index].export_exec.get();
        placement.nonexec_export = OP_set_halos_d[dat->set->index].export_nonexec.get();
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
    ensure_capacity(&gather_buf, &gather_buf_size, gather_size, false);
    ensure_capacity(&scatter_buf, &scatter_buf_size, scatter_size, false);

    gather_used = gather_size;
    scatter_used = scatter_size;

    if (OP_gpu_direct) {
        return {gather_buf, scatter_buf, gather_buf, scatter_buf};
    }

    ensure_host_capacity(&gather_host, &gather_host_size, gather_size);
    ensure_host_capacity(&scatter_host, &scatter_host_size, scatter_size);

    return {gather_buf, scatter_buf, gather_host, scatter_host};
}

// The specs carry everything the copy needs, so one kernel covers both
// directions. It stays templated on the spec type rather than taking a flag, so
// gather and scatter remain separate instantiations and separate rows in a
// profile.
template<typename SpecsT, typename DispsT>
__global__ void halo_copy_kernel(__grid_constant__ const SpecsT specs,
                                 __grid_constant__ const DispsT disps,
                                 __grid_constant__ const int num_specs) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    int lb = cub::LowerBound(disps, num_specs, thread_id + 1);
    if (lb == num_specs) return;

    auto index = lb > 0 ? thread_id - disps[lb - 1] : thread_id;

    copy_element(specs[lb], index);
}

// Specs small enough to ride along as kernel parameters.
template<unsigned N, typename SpecT>
static void launch_inline(const SpecsByNeighbour<SpecT> &specs_for_neighbour) {
    std::array<SpecT, N> specs;
    std::array<int, N> disps;

    int num_specs = 0;
    size_t total_size = 0;
    for (auto &[neighbour, batch] : specs_for_neighbour) {
        for (auto &spec : batch) {
            specs[num_specs] = spec;

            total_size += spec.size;
            disps[num_specs] = total_size;

            ++num_specs;
        }
    }

    size_t num_blocks = (total_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    if (num_blocks == 0) return;

    halo_copy_kernel<<<num_blocks, BLOCK_SIZE>>>(specs, disps, num_specs);
}

// Too many to pass by value, so stage them through device memory.
template<typename SpecT>
static void launch_staged(const SpecsByNeighbour<SpecT> &specs_for_neighbour,
                          SpecT **specs_d, size_t *specs_capacity,
                          int **disps_d, size_t *disps_capacity) {
    std::vector<SpecT> specs;
    std::vector<int> disps;

    size_t total_size = 0;
    for (auto &[neighbour, batch] : specs_for_neighbour) {
        for (auto &spec : batch) {
            specs.push_back(spec);

            total_size += spec.size;
            disps.push_back(total_size);
        }
    }

    ensure_capacity((void **) specs_d, specs_capacity, sizeof(SpecT) * specs.size());
    cutilSafeCall(gpuMemcpyAsync((void *) *specs_d, (void *) specs.data(),
                                 sizeof(SpecT) * specs.size(), gpuMemcpyHostToDevice));

    ensure_capacity((void **) disps_d, disps_capacity, sizeof(int) * disps.size());
    cutilSafeCall(gpuMemcpyAsync((void *) *disps_d, (void *) disps.data(),
                                 sizeof(int) * disps.size(), gpuMemcpyHostToDevice));

    size_t num_blocks = (total_size + (BLOCK_SIZE - 1)) / BLOCK_SIZE;
    if (num_blocks == 0) return;

    halo_copy_kernel<<<num_blocks, BLOCK_SIZE>>>(*specs_d, *disps_d, (int) specs.size());
}

template<typename SpecT>
static void launch_copies(const SpecsByNeighbour<SpecT> &specs_for_neighbour,
                          SpecT **specs_d, size_t *specs_capacity,
                          int **disps_d, size_t *disps_capacity) {
    size_t num_specs = 0;
    for (auto &[neighbour, batch] : specs_for_neighbour) {
        num_specs += batch.size();
    }

    if (num_specs == 0) return;

    if      (num_specs <= 4)  launch_inline<4>(specs_for_neighbour);
    else if (num_specs <= 8)  launch_inline<8>(specs_for_neighbour);
    else if (num_specs <= 16) launch_inline<16>(specs_for_neighbour);
    else if (num_specs <= 32) launch_inline<32>(specs_for_neighbour);
    else if (num_specs <= max_inline_specs) launch_inline<max_inline_specs>(specs_for_neighbour);
    else launch_staged(specs_for_neighbour, specs_d, specs_capacity, disps_d, disps_capacity);
}

static void record_event(gpuEvent_t &event, bool &initialised) {
    if (!initialised) {
        cutilSafeCall(gpuEventCreateWithFlags(&event, gpuEventDisableTiming));
        initialised = true;
    }

    cutilSafeCall(gpuEventRecord(event, 0));
}

void CudaBackend::initiate_gathers(const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) {
    launch_copies(gathers_for_neighbour, &gathers_d, &gathers_size,
                  &gather_disps_d, &gather_disps_size);

    // Staged: the send buffer has to reach the host before the sends go out, and
    // wait_gathers() covers this because it is enqueued before the event.
    if (!OP_gpu_direct && gather_used > 0) {
        cutilSafeCall(gpuMemcpyAsync(gather_host, gather_buf, gather_used,
                                     gpuMemcpyDeviceToHost, 0));
    }

    record_event(gather_event, gather_event_initialised);
}

void CudaBackend::initiate_scatters(const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) {
    // Staged: the received data is in host memory, so it has to reach the device
    // before the scatter kernel reads it.
    if (!OP_gpu_direct && scatter_used > 0) {
        cutilSafeCall(gpuMemcpyAsync(scatter_buf, scatter_host, scatter_used,
                                     gpuMemcpyHostToDevice, 0));
    }

    launch_copies(scatters_for_neighbour, &scatters_d, &scatters_size,
                  &scatter_disps_d, &scatter_disps_size);

    record_event(scatter_event, scatter_event_initialised);
}

void CudaBackend::wait_gathers() {
    cutilSafeCall(gpuEventSynchronize(gather_event));
}

void CudaBackend::wait_scatters() {
    if (!scatter_event_initialised) {
        return;
    }

    cutilSafeCall(gpuEventSynchronize(scatter_event));
}

}

namespace op::unified_exchanges {

// --- Asynchronous exchanges ---

static bool host_fits(size_t size, size_t capacity) { return OP_gpu_direct || size <= capacity; }

bool CudaBackend::grows(int slot, size_t gather_size, size_t scatter_size) const {
    const auto &s = async_slots[slot];
    return !host_fits(gather_size, s.gather_host_size) || !host_fits(scatter_size, s.scatter_host_size);
}

ExchangeBuffers CudaBackend::slot_buffers(int slot, size_t gather_size, size_t scatter_size) {
    auto &s = async_slots[slot];
    // Stream-ordered: the slot is free, so nothing still reads these.
    ensure_capacity(&s.gather_buf, &s.gather_buf_size, gather_size);
    ensure_capacity(&s.scatter_buf, &s.scatter_buf_size, scatter_size);
    s.gather_used = gather_size;
    s.scatter_used = scatter_size;

    if (OP_gpu_direct) {
        return {s.gather_buf, s.scatter_buf, s.gather_buf, s.scatter_buf};
    }

    ensure_host_capacity(&s.gather_host, &s.gather_host_size, gather_size);
    ensure_host_capacity(&s.scatter_host, &s.scatter_host_size, scatter_size);
    return {s.gather_buf, s.scatter_buf, s.gather_host, s.scatter_host};
}

void CudaBackend::slot_gathers(int slot, const SpecsByNeighbour<GatherSpec> &gathers_for_neighbour) {
    auto &s = async_slots[slot];
    launch_copies(gathers_for_neighbour, &gathers_d, &gathers_size,
                  &gather_disps_d, &gather_disps_size);
    if (!OP_gpu_direct && s.gather_used > 0) {
        cutilSafeCall(gpuMemcpyAsync(s.gather_host, s.gather_buf, s.gather_used,
                                     gpuMemcpyDeviceToHost, 0));
    }
    record_event(s.gather_event, s.gather_event_initialised);
}

void CudaBackend::slot_scatters(int slot, unsigned seq,
                                const SpecsByNeighbour<ScatterSpec> &scatters_for_neighbour) {
    if (flags == nullptr) {
        void *host = nullptr;
        cutilSafeCall(gpuHostAlloc(&host, slots * sizeof(unsigned), gpuHostAllocMapped));
        std::memset(host, 0, slots * sizeof(unsigned));
        cutilSafeCall(gpuHostGetDevicePointer(&flags_d, host, 0));
        flags = static_cast<volatile unsigned *>(host);
    }

    auto &s = async_slots[slot];
    auto result = gpuStreamWaitValueGeq32(0, static_cast<unsigned *>(flags_d) + slot, seq);
    if (result != GPU_SUCCESS) {
        std::fprintf(stderr, "OP2: the device cannot wait on a value in a stream (error %d), "
                             "which OP_MPI_ASYNC needs\n", static_cast<int>(result));
        std::exit(1);
    }
    if (!OP_gpu_direct && s.scatter_used > 0) {
        cutilSafeCall(gpuMemcpyAsync(s.scatter_buf, s.scatter_host, s.scatter_used,
                                     gpuMemcpyHostToDevice, 0));
    }
    launch_copies(scatters_for_neighbour, &scatters_d, &scatters_size,
                  &scatter_disps_d, &scatter_disps_size);
    record_event(s.scatter_event, s.scatter_event_initialised);
}

static bool event_done(gpuEvent_t event, bool initialised) {
    if (!initialised) return true;
    auto result = gpuEventQuery(event);
    if (result == gpuErrorNotReady) return false;
    cutilSafeCall(result);
    return true;
}

bool CudaBackend::gathered(int slot) {
    return event_done(async_slots[slot].gather_event, async_slots[slot].gather_event_initialised);
}

bool CudaBackend::scattered(int slot) {
    return event_done(async_slots[slot].scatter_event, async_slots[slot].scatter_event_initialised);
}

void CudaBackend::raise(int slot, unsigned seq) {
    // The receives are complete; the device reads the flag across PCIe.
    __atomic_thread_fence(__ATOMIC_RELEASE);
    flags[slot] = seq;
}

}
