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

enum class HierScalarType {
    f32,
    f64,
    i32,
};

struct HierArgDescriptor {
    int arg_index;
    int dat_index;
};

// The indirect arguments a hierarchical strategy plans over, grouped by dat:
// the increments for hierarchical atomics, every argument on an incremented
// or read-written dat for hierarchical colouring.  Generated with the wrapper
// it describes.
struct HierArgGroups {
    std::span<const HierArgDescriptor> args;
    // Scalar type of each dat group, in shared-region order.
    std::span<const HierScalarType> dats;
    int chunk_size_override = -1;
};

// Why a strategy passed a loop down the ladder.  These are stable diagnostic
// names, reported once per loop outcome or plan at OP_diags > 3.
enum class FallbackReason {
    none,
    disabled,             // the strategy's variable turns it off
    // Plan outcomes, decided once per cache key.
    no_active_argument,
    incompatible_argument,
    insufficient_shared_memory,
    low_compression,      // below OP_HIER_ATOMICS_MIN_COMPRESSION
    too_many_colours,     // a chunk needs more thread colours than fit a byte
};

// Convert a fallback reason to its stable diagnostic name.
std::string_view
fallback_reason_name(FallbackReason reason);

struct HierPlanOptions {
    int block_size = 128;
    int requested_chunk_size = 0;
    std::size_t shared_memory_limit = 0;
    // Marking exclusive owners changes the packed words, so a plan built with
    // it off is a different plan and is keyed separately.
    bool exclusive_flush = true;
};

struct HierPlanStatistics {
    std::size_t raw_references = 0;
    std::size_t distinct_targets = 0;
    // Staging: owners whose target occurs in exactly one chunk of its
    // section, so the flush needs no global atomic.  The rest of
    // distinct_targets flush with atomicAdd.
    std::size_t exclusive_owners = 0;
    // Colouring: launches over all sections, and the most thread colours any
    // chunk needs.
    std::size_t launches = 0;
    int max_thread_colours = 0;

    // Return the staged references per global flush.
    double compression() const {
        return distinct_targets > 0
                   ? static_cast<double>(raw_references) /
                         static_cast<double>(distinct_targets)
                   : 0.0;
    }
};

// Shared-memory slots for every staged reference.
struct HierStaging {
    // Lets the staged wrapper skip its seed pass, which would otherwise read
    // every packed word back from global memory to find nothing.
    int has_exclusive = 0;

    std::vector<HierSmemStageWord> stage_words;
    std::vector<int> stage_counts;
    std::vector<std::size_t> section_shared_bytes;
};

// Block and thread colours.  Each section runs one launch per block colour;
// chunk_order lists the chunks launch by launch.
struct HierColouring {
    std::vector<int> chunk_order;
    std::vector<int> launch_chunk_offsets;
    std::vector<int> section_launch_offsets;
    std::vector<std::uint8_t> thread_colours;
    std::vector<int> chunk_thread_colours;
};

// Consecutive source chunks inside each schedule section, plus whatever the
// strategy layers over them.
struct HierPlan {
    int selected_chunk_size = 0;
    int set_stride = 0;

    std::vector<int> source_offsets;
    std::vector<int> section_chunk_offsets;

    std::optional<HierStaging> staging;
    std::optional<HierColouring> colouring;

    HierPlanStatistics statistics;

    // Return the number of consecutive source chunks in the plan.
    std::size_t num_chunks() const {
        return source_offsets.empty() ? 0 : source_offsets.size() - 1;
    }
};

struct HierPlanBuildResult {
    FallbackReason reason = FallbackReason::none;
    std::optional<HierPlan> plan;

    // Report whether planning succeeded and produced a usable plan.
    explicit operator bool() const {
        return reason == FallbackReason::none && plan.has_value();
    }
};

// Build the largest block-aligned staging plan that fits the supplied byte
// limit.
HierPlanBuildResult build_hier_atomics_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierArgGroups& descriptor,
    const HierPlanOptions& options);

// Build a colouring plan: chunks of each section coloured so that no two chunks
// of a colour, and no two elements of a chunk and thread colour, reach the same
// target of a grouped argument.
HierPlanBuildResult build_hier_colouring_plan(
    op_set set, std::span<const op_arg> args,
    std::span<const ExecutionSection> sections,
    const HierArgGroups& descriptor,
    const HierPlanOptions& options);

using HierPlanReleaseCallback = void (*)(void *owner);

// Register owners of plan caches and JIT compiles for explicit backend
// shutdown, which op_exit runs before tearing the device down.
void register_hier_plan_owner(void *owner,
                              HierPlanReleaseCallback release);
void unregister_hier_plan_owner(void *owner);
void release_hier_plan_device_storage();

} // namespace op::f2c
