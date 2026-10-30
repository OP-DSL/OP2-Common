#pragma once

// A chunk plan as the GPU kernels read it: its arrays resident on the device,
// which a chunk launch's record carries (op2/rt/abi/gpu/launch.h), plus the
// per-call wavefront state.  Defined in src/rt/gpu/op_gpu_chunk_encode.cpp.

#include <op2/rt/abi/gpu/launch.h>
#include <op2/rt/abi/staging.h>
#include <op2/rt/gpu/chunk_plan.h>

#include <cstddef>
#include <cstdint>

namespace op::rt::gpu {

struct ChunkArrays {
    int *source_offsets = nullptr;
    abi::StageWord *stage_words = nullptr;
    int *stage_counts = nullptr;
    int *slot_offsets = nullptr;
    int *slot_targets = nullptr;
    int *chunk_order = nullptr;
    std::uint8_t *thread_colours = nullptr;
    int *chunk_thread_colours = nullptr;
    int *wave_pred_offsets = nullptr;
    int *wave_preds = nullptr;
    int *wave_done = nullptr;
    int *wave_tickets = nullptr;
};

class DevicePlan : public ChunkArrays {
private:
    // Calls are ordered on one stream, so each needs only a fresh epoch.
    mutable int m_wave_epoch = 0;
    std::size_t m_wave_positions = 0;

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
    int next_wave_epoch() const;

    // OP2 selects one device at initialisation and never switches, so these
    // free on the device they were allocated on.
    void release();

    // A chunk launch's record carries the arrays.
    void fill(abi::gpu::Chunk& record) const;
};

} // namespace op::rt::gpu
