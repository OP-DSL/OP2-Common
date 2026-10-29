#pragma once

// A chunk plan as the GPU kernels read it: its arrays resident on the device
// (op2/rt/gpu/loop.h's ChunkArrays), plus the per-call wavefront state.
// Defined in src/rt/gpu/op_gpu_chunk_encode.cpp.

#include <op2/rt/gpu/chunk_plan.h>
#include <op2/rt/gpu/loop.h>

namespace op::rt::gpu {

class DevicePlan : public ChunkArrays {
private:
    // Calls are ordered on one stream, so each needs only a fresh epoch.
    mutable int m_wave_epoch = 0;

public:
    DevicePlan() = default;
    DevicePlan(const DevicePlan&) = delete;
    DevicePlan& operator=(const DevicePlan&) = delete;

    DevicePlan(DevicePlan&& other) noexcept { *this = std::move(other); }
    DevicePlan& operator=(DevicePlan&& other) noexcept;

    // Upload the arrays a staged kernel reads.
    explicit DevicePlan(const ChunkPlan& plan);

    ~DevicePlan() { release(); }

    // The epoch a wavefront call marks its positions done with.
    int next_wave_epoch() const { return ++m_wave_epoch; }

    // OP2 selects one device at initialisation and never switches, so these
    // free on the device they were allocated on.
    void release();
};

} // namespace op::rt::gpu
