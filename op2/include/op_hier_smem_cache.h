#pragma once

#include <op_gpu_shims.h>
#include <op_hier_smem_plan.h>

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace op::f2c {

struct HierSmemPlanDeviceView {
    const int *source_offsets = nullptr;
    const HierSmemStageWord *stage_words = nullptr;
    const int *stage_counts = nullptr;
};

struct HierSmemPlanCacheStatistics {
    std::size_t entries = 0;
    std::size_t builds = 0;
    std::size_t uploads = 0;
};

namespace detail {

// One loop's plan varies only with optional state and the runtime objects
// behind its arguments; the rest of the schema is fixed by the KernelInfo this
// cache belongs to.  OP2 runs on one device for the process lifetime, so the
// device is not part of the key either.
//
// Every dat argument contributes its active dat, not just the staged ones:
// eligibility and the aliasing checks both read arguments outside the staging
// descriptor, so activating one of those has to miss the cache.
struct HierSmemPlanKey {
    int set_index = -1;
    int section_count = 0;
    int block_size = 0;
    int requested_chunk_size = 0;
    // Active dat index per argument, then map and component per staged one.
    std::vector<int> dats;
    std::vector<int> staged;

    bool operator==(const HierSmemPlanKey&) const = default;
};

inline HierSmemPlanKey make_hier_smem_plan_key(
    op_set set, std::span<const op_arg> args, int section_count,
    const HierSmemStagingDescriptor& descriptor,
    const HierSmemPlanOptions& options) {
    assert(set != nullptr && section_count > 0);
    HierSmemPlanKey key;
    key.set_index = set->index;
    key.section_count = section_count;
    key.block_size = options.block_size;
    key.requested_chunk_size = options.requested_chunk_size;

    key.dats.assign(args.size(), -1);
    for (std::size_t i = 0; i < args.size(); ++i) {
        const op_arg& arg = args[i];
        if (arg.opt != 0 && arg.argtype == OP_ARG_DAT && arg.dat != nullptr)
            key.dats[i] = arg.dat->index;
    }

    key.staged.reserve(2 * descriptor.args.size());
    for (const auto& descriptor_arg : descriptor.args) {
        auto index = static_cast<std::size_t>(descriptor_arg.arg_index);
        assert(index < args.size());
        const op_arg& arg = args[index];

        if (arg.opt == 0 || arg.map == nullptr) {
            key.staged.insert(key.staged.end(), {-1, -1});
            continue;
        }

        key.staged.insert(key.staged.end(), {arg.map->index, arg.idx});
    }

    return key;
}

inline void hier_smem_check_gpu(gpuError_t result, const char *operation) {
    if (result == gpuSuccess)
        return;

    std::fprintf(stderr, "error: %s failed with %s\n", operation,
                 gpuGetErrorString(result));
    std::exit(EXIT_FAILURE);
}

class HierSmemPlanCacheEntry {
private:
    HierSmemFallbackReason m_reason;
    std::optional<HierSmemPlan> m_plan;
    int *m_source_offsets_d = nullptr;
    HierSmemStageWord *m_stage_words_d = nullptr;
    int *m_stage_counts_d = nullptr;

    template<typename T>
    // Upload only the compact arrays read by a staged wrapper.
    void upload(const std::vector<T>& source, T *&destination) {
        if (source.empty())
            return;

        hier_smem_check_gpu(
            gpuMalloc(reinterpret_cast<void **>(&destination),
                      source.size() * sizeof(T)),
            "gpuMalloc");
        hier_smem_check_gpu(
            gpuMemcpy(destination, source.data(), source.size() * sizeof(T),
                      gpuMemcpyHostToDevice),
            "gpuMemcpy");
    }

public:
    HierSmemPlanCacheEntry(const HierSmemPlanCacheEntry&) = delete;

    explicit HierSmemPlanCacheEntry(HierSmemPlanBuildResult result)
        : m_reason{result.reason}, m_plan{std::move(result.plan)} {
        if (!m_plan.has_value())
            return;

        upload(m_plan->source_offsets, m_source_offsets_d);
        upload(m_plan->stage_words, m_stage_words_d);
        upload(m_plan->stage_counts, m_stage_counts_d);
    }

    ~HierSmemPlanCacheEntry() { release_device_storage(); }

    explicit operator bool() const { return m_plan.has_value(); }
    HierSmemFallbackReason reason() const { return m_reason; }
    const HierSmemPlan *plan() const {
        return m_plan.has_value() ? &*m_plan : nullptr;
    }
    HierSmemPlanDeviceView device_view() const {
        return {m_source_offsets_d, m_stage_words_d, m_stage_counts_d};
    }

    // OP2 selects one device at initialization and never switches, so these
    // free on the device they were allocated on.
    void release_device_storage() {
        if (m_source_offsets_d != nullptr)
            hier_smem_check_gpu(gpuFree(m_source_offsets_d), "gpuFree");
        if (m_stage_words_d != nullptr)
            hier_smem_check_gpu(gpuFree(m_stage_words_d), "gpuFree");
        if (m_stage_counts_d != nullptr)
            hier_smem_check_gpu(gpuFree(m_stage_counts_d), "gpuFree");

        m_source_offsets_d = nullptr;
        m_stage_words_d = nullptr;
        m_stage_counts_d = nullptr;
    }
};

// Cache both usable plans and fallback results; only usable plans own device
// copies of the compact arrays consumed by staged wrappers.
// Cache both usable plans and fallback results; only usable plans own device
// copies of the compact arrays consumed by staged wrappers.  A loop sees one
// key in the common case and a second when optional state flips, so the
// entries are scanned linearly rather than hashed.
class HierSmemPlanCache {
private:
    std::vector<std::pair<HierSmemPlanKey,
                          std::unique_ptr<HierSmemPlanCacheEntry>>>
        m_entries;
    std::size_t m_builds = 0;
    std::size_t m_uploads = 0;

public:
    template<typename Builder>
    // Build and upload once for each exact runtime configuration.
    const HierSmemPlanCacheEntry& get_or_build(HierSmemPlanKey key,
                                               Builder&& builder) {
        for (const auto& [existing, entry] : m_entries)
            if (existing == key)
                return *entry;

        ++m_builds;
        auto entry = std::make_unique<HierSmemPlanCacheEntry>(
            std::forward<Builder>(builder)());
        if (*entry)
            ++m_uploads;

        m_entries.emplace_back(std::move(key), std::move(entry));
        return *m_entries.back().second;
    }

    HierSmemPlanCacheStatistics statistics() const {
        return {m_entries.size(), m_builds, m_uploads};
    }

    void clear() { m_entries.clear(); }
};

} // namespace detail
} // namespace op::f2c
