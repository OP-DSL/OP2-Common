#pragma once

#include <op_f2c_prelude.h>
#include <op_lib_core.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace op::f2c {

struct ExecutionSection {
    int start;
    int end;

    // Return the number of source elements in this launch section.
    int size() const { return end - start; }
};

enum class HierSmemScalarType {
    f32,
    f64,
    i32,
};

struct HierSmemArgDescriptor {
    int arg_index;
    int dat_index;
};

struct HierSmemDatDescriptor {
    HierSmemScalarType scalar_type;
};

struct HierSmemStagingDescriptor {
    std::span<const HierSmemArgDescriptor> args;
    std::span<const HierSmemDatDescriptor> dats;
    int chunk_size_override = -1;
};

enum class HierSmemFallbackReason {
    none,
    no_active_increment,
    incompatible_argument,
    insufficient_shared_memory,
};

// Convert a fallback reason to its stable diagnostic name.
std::string_view
hier_smem_fallback_reason_name(HierSmemFallbackReason reason);

struct HierSmemPlanOptions {
    int block_size = 128;
    int requested_chunk_size = 0;
    std::size_t shared_memory_limit = 0;
};

struct HierSmemPlanStatistics {
    std::size_t raw_references = 0;
    std::size_t distinct_targets = 0;
};

struct HierSmemPlan {
    int selected_chunk_size = 0;
    int set_stride = 0;

    std::vector<int> source_offsets;
    std::vector<int> section_chunk_offsets;
    std::vector<HierSmemStageWord> stage_words;
    std::vector<int> stage_counts;
    std::vector<std::size_t> section_shared_bytes;

    HierSmemPlanStatistics statistics;

    // Return the number of consecutive source chunks in the plan.
    std::size_t num_chunks() const {
        return source_offsets.empty() ? 0 : source_offsets.size() - 1;
    }
};

struct HierSmemPlanBuildResult {
    HierSmemFallbackReason reason = HierSmemFallbackReason::none;
    std::optional<HierSmemPlan> plan;

    // Report whether planning succeeded and produced a usable plan.
    explicit operator bool() const {
        return reason == HierSmemFallbackReason::none && plan.has_value();
    }
};

// Build the largest block-aligned plan that fits the supplied byte limit.
HierSmemPlanBuildResult build_hier_smem_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierSmemStagingDescriptor& descriptor,
    const HierSmemPlanOptions& options);

using HierSmemPlanReleaseCallback = void (*)(void *owner);

// Register device-owning plan caches for explicit backend shutdown.
void register_hier_smem_plan_owner(void *owner,
                                   HierSmemPlanReleaseCallback release);
void unregister_hier_smem_plan_owner(void *owner);
void release_hier_smem_plan_device_storage();

} // namespace op::f2c
