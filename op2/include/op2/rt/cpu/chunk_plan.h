#pragma once

// The OpenMP backend's chunk plans, built from the plan building blocks: a
// thread runs a chunk on its own, so a chunk needs no lanes, and a staged
// chunk's scratch is bounded only so its slots stay in the caches.  The
// planners are defined in src/rt/cpu/op_cpu_chunk_plan.cpp.

#include <op2/rt/abi/staging.h>
#include <op2/rt/plan/chunks.h>
#include <op2/rt/plan/staging.h>

#include <cstddef>
#include <span>
#include <vector>

namespace op::rt::cpu {

// Elements per chunk unless OP_CHUNK_SIZES says otherwise.
constexpr int default_chunk_size = 256;

// The scratch a staged chunk may use.
constexpr std::size_t chunk_scratch_limit = std::size_t{1} << 20;

// Each batch's chunks shared out over the threads.  chunk_atomics stages
// each chunk's slots in its thread's scratch, a 16-bit stage word each as
// the region reads them.
struct ChunkPlan : plan::ChunkPlan<abi::StageWord> {
    // The most scratch bytes a chunk needs.
    std::size_t max_scratch_bytes = 0;
};

// A rejection's reason is one of plan::stage_chunks' or
// plan::updated_references', or low_compression.
using ChunkResult = plan::PlanResult<ChunkPlan>;

// chunk_atomics: a section's chunks run at once and may share targets, so a
// chunk accumulates into its slots and flushes them atomically.  A plan that
// barely combines references is rejected.
ChunkResult plan_chunk_atomics(std::span<const op_arg> args,
                               const op::rt::loop::Sections& sections,
                               const abi::ArgGroups& groups, int size,
                               double min_compression);

// chunk_colouring: the chunks of a batch share no target, so a chunk runs
// the plain body over its elements.
ChunkResult plan_chunk_colouring(std::span<const op_arg> args,
                                 const op::rt::loop::Sections& sections, int size);

} // namespace op::rt::cpu
