#pragma once

// A chunk plan as the GPU kernels read it: its arrays resident on the device,
// which a chunk launch's record carries (op2/rt/abi/gpu/launch.h), plus the
// per-call wavefront state.  Defined in src/rt/gpu/op_gpu_chunk_encode.cpp.

#include <op2/rt/abi/gpu/launch.h>
#include <op2/rt/gpu/chunk_plan.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace op::rt::gpu {

class DevicePlan {
private:
    // The record's plan arrays, as allocated.
#define OP_GPU_PLAN_ARRAY(type, name, qualifier)                                         \
    std::remove_const_t<std::remove_pointer_t<type>> *m_##name = nullptr;
    OP_ABI_GPU_CHUNK_ARRAYS(OP_GPU_PLAN_ARRAY)
#undef OP_GPU_PLAN_ARRAY
    // A wavefront plan's next position per section, and its positions and
    // sections; calls are ordered on one stream, so each needs only a fresh
    // epoch.
    int *m_wave_tickets = nullptr;
    std::size_t m_wave_positions = 0;
    std::size_t m_wave_sections = 0;
    mutable int m_wave_epoch = 0;

    // OP2 selects one device at initialisation and never switches, so this
    // frees on the device the arrays were allocated on.
    void free_device();

public:
    DevicePlan() = default;
    DevicePlan(const DevicePlan&) = delete;
    DevicePlan& operator=(const DevicePlan&) = delete;

    DevicePlan(DevicePlan&& other) noexcept { *this = std::move(other); }
    DevicePlan& operator=(DevicePlan&& other) noexcept;

    // Upload the arrays a staged kernel reads.
    explicit DevicePlan(const ChunkPlan& plan);

    ~DevicePlan() { free_device(); }

    // Before a call's launches of a wavefront plan: every section's ticket
    // from zero, and a fresh epoch to mark its positions done with.
    void begin_wavefront_call() const;

    // A chunk launch's record of section: the arrays, and a wavefront plan's
    // ticket and epoch.
    void fill(abi::gpu::Chunk& record, int section) const;
};

} // namespace op::rt::gpu
