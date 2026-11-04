#include <op2/rt/gpu/chunk_encode.h>

#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/gpu/upload.h>

#include <cstddef>
#include <cstdint>
#include <limits>
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

int *allocate_zeroed(std::size_t count) {
    int *destination = nullptr;
    GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&destination), count * sizeof(int)));
    GPU_CHECK(gpuMemset(destination, 0, count * sizeof(int)));
    return destination;
}

} // namespace

DevicePlan::DevicePlan(const ChunkPlan& plan) {
    upload(plan.chunks.offsets, m_chunk_offsets);
    upload(plan.stage_words, m_stage_words);
    upload(slot_counts(plan.staging), m_stage_counts);
    upload(plan.staging.slot_offsets, m_slot_offsets);
    upload(plan.staging.slot_targets, m_slot_targets);
    upload(plan.batches.order, m_chunk_order);
    upload(plan.thread_colours, m_thread_colours);
    upload(plan.chunk_thread_colours, m_chunk_thread_colours);
    if (plan.wavefront) {
        upload(plan.wave_pred_offsets, m_wave_pred_offsets);
        upload(plan.wave_preds, m_wave_preds);
        m_wave_positions = plan.batches.order.size();
        m_wave_sections = plan.batches.section_batches.size() - 1;
        m_wave_done = allocate_zeroed(m_wave_positions);
        m_wave_tickets = allocate_zeroed(m_wave_sections);
    }
}

DevicePlan& DevicePlan::operator=(DevicePlan&& other) noexcept {
    if (this != &other) {
        free_device();
#define OP_GPU_PLAN_SWAP(type, name, qualifier) std::swap(m_##name, other.m_##name);
        OP_ABI_GPU_CHUNK_ARRAYS(OP_GPU_PLAN_SWAP)
#undef OP_GPU_PLAN_SWAP
        std::swap(m_wave_tickets, other.m_wave_tickets);
        std::swap(m_wave_positions, other.m_wave_positions);
        std::swap(m_wave_sections, other.m_wave_sections);
        std::swap(m_wave_epoch, other.m_wave_epoch);
    }
    return *this;
}

void DevicePlan::begin_wavefront_call() const {
    if (m_wave_tickets == nullptr)
        return;
    // The done flags hold earlier calls' epochs: before the epoch overflows,
    // clear them and count from one again.
    if (m_wave_epoch == std::numeric_limits<int>::max()) {
        GPU_CHECK(gpuMemsetAsync(m_wave_done, 0, m_wave_positions * sizeof(int), 0));
        m_wave_epoch = 0;
    }
    ++m_wave_epoch;
    GPU_CHECK(gpuMemsetAsync(m_wave_tickets, 0, m_wave_sections * sizeof(int), 0));
}

void DevicePlan::fill(abi::gpu::Chunk& record, int section) const {
#define OP_GPU_PLAN_FILL(type, name, qualifier) record.name = m_##name;
    OP_ABI_GPU_CHUNK_ARRAYS(OP_GPU_PLAN_FILL)
#undef OP_GPU_PLAN_FILL
    if (m_wave_tickets != nullptr) {
        record.wave_ticket = m_wave_tickets + section;
        record.wave_epoch = m_wave_epoch;
    }
}

void DevicePlan::free_device() {
#define OP_GPU_PLAN_FREE(type, name, qualifier) release(m_##name);
    OP_ABI_GPU_CHUNK_ARRAYS(OP_GPU_PLAN_FREE)
#undef OP_GPU_PLAN_FREE
    release(m_wave_tickets);
}

} // namespace op::rt::gpu
