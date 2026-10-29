#include <op2/rt/gpu/chunk_encode.h>

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/gpu/upload.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace op::rt::gpu {

namespace {

std::vector<int> slot_counts(const plan::Staging& staging) {
    std::vector<int> counts;
    for (std::size_t region = 0; region + 1 < staging.slot_offsets.size(); ++region)
        counts.push_back(staging.slot_offsets[region + 1] - staging.slot_offsets[region]);
    return counts;
}

void allocate_zeroed(std::size_t count, int *&destination) {
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&destination), count * sizeof(int)));
    GPU_CHECK(gpuMemset(destination, 0, count * sizeof(int)));
}

} // namespace

void DevicePlan::fill(abi::gpu::Chunk& record) const {
    record.source_offsets = source_offsets;
    record.stage_words = stage_words;
    record.stage_counts = stage_counts;
    record.slot_offsets = slot_offsets;
    record.slot_targets = slot_targets;
    record.chunk_order = chunk_order;
    record.thread_colours = thread_colours;
    record.chunk_thread_colours = chunk_thread_colours;
    record.wave_pred_offsets = wave_pred_offsets;
    record.wave_preds = wave_preds;
    record.wave_done = wave_done;
}

DevicePlan& DevicePlan::operator=(DevicePlan&& other) noexcept {
    if (this != &other) {
        release();
        std::swap(static_cast<ChunkArrays&>(*this), static_cast<ChunkArrays&>(other));
        std::swap(m_wave_epoch, other.m_wave_epoch);
    }
    return *this;
}

DevicePlan::DevicePlan(const ChunkPlan& plan) {
    upload(plan.chunks.offsets, source_offsets);
    upload(plan.stage_words, stage_words);
    upload(slot_counts(plan.staging), stage_counts);
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

void DevicePlan::release() {
    gpu::release(source_offsets);
    gpu::release(stage_words);
    gpu::release(stage_counts);
    gpu::release(slot_offsets);
    gpu::release(slot_targets);
    gpu::release(chunk_order);
    gpu::release(thread_colours);
    gpu::release(chunk_thread_colours);
    gpu::release(wave_pred_offsets);
    gpu::release(wave_preds);
    gpu::release(wave_done);
    gpu::release(wave_tickets);
}

} // namespace op::rt::gpu
