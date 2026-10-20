#pragma once

// A chunk plan as the GPU kernels read it: its arrays resident on the device
// through op::device(), plus the per-call wavefront state.  Host code shared
// by the CUDA and HIP strategies, and testable against the host device.

#include <gpu/chunk_plan.h>
#include <op_device.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace op::gpu {

class DevicePlan {
private:
    // Calls are ordered on one stream, so each needs only a fresh epoch.
    mutable int m_wave_epoch = 0;

    template<typename T>
    static void upload(const std::vector<T>& source, T *&destination) {
        if (source.empty())
            return;
        const auto& device = op::device();
        destination = static_cast<T *>(device.malloc(source.size() * sizeof(T)));
        device.copy(destination, source.data(), source.size() * sizeof(T));
    }

    static void allocate_zeroed(std::size_t count, int *&destination) {
        const auto& device = op::device();
        destination = static_cast<int *>(device.malloc(count * sizeof(int)));
        device.memset(destination, 0, count * sizeof(int));
    }

    template<typename T>
    static void release(T *&pointer) {
        if (pointer != nullptr)
            op::device().free(pointer);
        pointer = nullptr;
    }

public:
    int *source_offsets = nullptr;
    plan::StageWord *stage_words = nullptr;
    int *stage_counts = nullptr;
    int *slot_offsets = nullptr;
    int *slot_targets = nullptr;
    int *chunk_order = nullptr;
    std::uint8_t *thread_colours = nullptr;
    int *chunk_thread_colours = nullptr;

    // Wavefront plans: predecessors, a done flag per position holding
    // the epoch of the call that last ran it, and a counter per section.
    int *wave_pred_offsets = nullptr;
    int *wave_preds = nullptr;
    int *wave_done = nullptr;
    int *wave_tickets = nullptr;

    DevicePlan() = default;
    DevicePlan(const DevicePlan&) = delete;
    DevicePlan& operator=(const DevicePlan&) = delete;

    DevicePlan(DevicePlan&& other) noexcept { *this = std::move(other); }
    DevicePlan& operator=(DevicePlan&& other) noexcept {
        if (this != &other) {
            release();
            std::swap(source_offsets, other.source_offsets);
            std::swap(stage_words, other.stage_words);
            std::swap(stage_counts, other.stage_counts);
            std::swap(slot_offsets, other.slot_offsets);
            std::swap(slot_targets, other.slot_targets);
            std::swap(chunk_order, other.chunk_order);
            std::swap(thread_colours, other.thread_colours);
            std::swap(chunk_thread_colours, other.chunk_thread_colours);
            std::swap(wave_pred_offsets, other.wave_pred_offsets);
            std::swap(wave_preds, other.wave_preds);
            std::swap(wave_done, other.wave_done);
            std::swap(wave_tickets, other.wave_tickets);
            std::swap(m_wave_epoch, other.m_wave_epoch);
        }
        return *this;
    }

    // Upload the arrays a staged kernel reads.
    explicit DevicePlan(const ChunkPlan& plan) {
        upload(plan.chunks.offsets, source_offsets);
        upload(plan.staging.stage_words, stage_words);
        upload(plan.staging.stage_counts, stage_counts);
        upload(plan.staging.slot_offsets, slot_offsets);
        upload(plan.staging.slot_targets, slot_targets);
        upload(plan.batches.order, chunk_order);
        upload(plan.thread_colours, thread_colours);
        upload(plan.chunk_thread_colours, chunk_thread_colours);
        if (plan.wavefront) {
            upload(plan.wave_pred_offsets, wave_pred_offsets);
            upload(plan.wave_preds, wave_preds);
            allocate_zeroed(plan.batches.order.size(), wave_done);
            allocate_zeroed(plan.batches.section_batches.size() - 1, wave_tickets);
        }
    }

    ~DevicePlan() { release(); }


    // The epoch a wavefront call marks its positions done with.
    int next_wave_epoch() const { return ++m_wave_epoch; }

    // OP2 selects one device at initialisation and never switches, so these
    // free on the device they were allocated on.
    void release() {
        release(source_offsets);
        release(stage_words);
        release(stage_counts);
        release(slot_offsets);
        release(slot_targets);
        release(chunk_order);
        release(thread_colours);
        release(chunk_thread_colours);
        release(wave_pred_offsets);
        release(wave_preds);
        release(wave_done);
        release(wave_tickets);
    }
};

} // namespace op::gpu
