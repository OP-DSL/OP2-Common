#pragma once

// The chunked strategies' building blocks: a loop's elements cut into chunks
// inside its sections, the targets each chunk reaches, batches of chunks
// that share none, a wavefront order over them, colours for a chunk's
// elements, and the result a backend's plan comes back in.  Host algorithms with no device knowledge and no device limits;
// a backend composes them into its plans and maps those onto its hardware.

#include <loop/strategy.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace op::plan {

// --- References -----------------------------------------------------------------

// A dat whose targets the planner follows, with what bounds them and how a
// staged region of them is laid out; dat is null while no argument of the
// group is active.
struct Group {
    op_dat dat = nullptr;
    int extent = 0;
    int dimension = 0;
    std::size_t scalar_size = 0;
};

// An indirect argument and the group of the dat it reaches.
struct Reference {
    const op_arg *arg;
    int group;
};

// The arguments the planner follows; an inactive one keeps its place and is
// skipped.
struct References {
    std::vector<Reference> items;
    std::vector<Group> groups;
};

// The indirect arguments a staged kernel stages, grouped by dat as it lays
// its regions out.  Generated with the kernel they describe.
struct ArgGroup {
    int arg;
    int group;
};

struct ArgGroups {
    std::span<const ArgGroup> args;
    int groups = 0;
};

// The groups' arguments as references, or why the call does not suit them:
// no_active_argument, or incompatible_argument when two groups reach one dat,
// or an argument outside the groups reaches a grouped dat or makes an update
// the groups would not order: an indirect increment, or with read_write an
// indirect read-write.
std::string_view grouped_references(std::span<const op_arg> args,
                                    const ArgGroups& groups, bool read_write,
                                    References& references);

// Every active indirect argument on a dat the loop increments or
// read-writes through a map, grouped by dat: what concurrent chunks must not
// share.  Rejected as no_active_argument when there is none, and as
// incompatible_argument when one reaches more than one map component.
std::string_view updated_references(std::span<const op_arg> args,
                                    References& references);

// --- Chunks ---------------------------------------------------------------------

// The elements of each section in consecutive chunks of up to size, so no
// chunk spans a halo wait or the global processing.
struct Chunks {
    int size = 0;
    // Per chunk its first element, then the last chunk's end.
    std::vector<int> offsets;
    // Per section its first chunk, then the chunk count.
    std::vector<int> section_offsets;

    int count() const {
        return offsets.empty() ? 0 : static_cast<int>(offsets.size()) - 1;
    }
    int begin(int chunk) const { return offsets[static_cast<std::size_t>(chunk)]; }
    int end(int chunk) const { return offsets[static_cast<std::size_t>(chunk) + 1]; }
};

Chunks chunk_sections(const op::loop::Sections& sections, int size);

// Each chunk's distinct targets of each group in ascending order, with the
// references the chunk makes to each.  Region chunk * groups + group.
struct ChunkTargets {
    int groups = 0;
    // Per group, the bound of its targets.
    std::vector<int> extents;
    // Per region its first entry, then the entry count.
    std::vector<int> offsets;
    std::vector<int> targets;
    std::vector<int> references;
    std::size_t total_references = 0;

    std::size_t region(int chunk, int group) const {
        return static_cast<std::size_t>(chunk) * static_cast<std::size_t>(groups) +
               static_cast<std::size_t>(group);
    }

    // f(group, target) for each of the chunk's distinct targets.
    template<typename F>
    void for_each(int chunk, F&& f) const {
        for (int group = 0; group < groups; ++group) {
            auto index = region(chunk, group);
            for (int entry = offsets[index]; entry < offsets[index + 1]; ++entry)
                f(static_cast<std::size_t>(group),
                  targets[static_cast<std::size_t>(entry)]);
        }
    }

    // References per distinct target.
    double compression() const {
        return targets.empty() ? 0.0
                               : static_cast<double>(total_references) /
                                     static_cast<double>(targets.size());
    }
};

ChunkTargets chunk_targets(const Chunks& chunks, const References& references);

// --- Batches --------------------------------------------------------------------

// Per section, batches of chunks run one after another, the chunks of a
// batch at once; a section without chunks has no batch.  order lists the
// chunks batch by batch; empty, the positions are the chunks themselves.
struct ChunkBatches {
    std::vector<int> order;
    // Per batch its first position, then the position count.
    std::vector<int> batch_offsets;
    // Per section its first batch, then the batch count.
    std::vector<int> section_batches;

    int chunk(int position) const {
        return order.empty() ? position : order[static_cast<std::size_t>(position)];
    }
    int count() const {
        return batch_offsets.empty() ? 0 : static_cast<int>(batch_offsets.size()) - 1;
    }
    int batches(int section) const {
        auto index = static_cast<std::size_t>(section);
        return section_batches[index + 1] - section_batches[index];
    }
    int batch_index(int section, int batch) const {
        return section_batches[static_cast<std::size_t>(section)] + batch;
    }
    std::pair<int, int> positions(int section, int batch) const {
        auto index = static_cast<std::size_t>(batch_index(section, batch));
        return {batch_offsets[index], batch_offsets[index + 1]};
    }
};

// Each section's chunks as one batch, in chunk order.
ChunkBatches section_batches(const Chunks& chunks);

// Each section's chunks coloured so that the chunks of a batch share no
// target, window chunks at a time; zero colours a section at once.
ChunkBatches colour_chunks(const Chunks& chunks, const ChunkTargets& targets,
                           int window = 0);

// Coloured batches run as one batch per section, in their order, a position
// once its predecessors are done: the latest earlier position of its section
// sharing each of its targets, which covers every earlier one sharing it.
struct Wavefront {
    ChunkBatches batches;
    // Per position its first predecessor, then the predecessor count.
    std::vector<int> pred_offsets;
    std::vector<int> preds;
};

Wavefront wavefront(const ChunkBatches& coloured, const ChunkTargets& targets);

// --- Plans ----------------------------------------------------------------------

// A backend's plan for a call, or why the call does not suit the strategy;
// a rejection has no plan, and a low_compression one keeps its compression.
template<typename Plan>
struct PlanResult {
    std::string_view rejected;
    std::unique_ptr<Plan> plan;
    double compression = 0.0;

    explicit operator bool() const { return rejected.empty() && plan != nullptr; }
};

// --- Element colours ------------------------------------------------------------

// Each chunk's elements coloured so that no two of a colour share a target:
// per element its colour within its chunk, per chunk its colour count.
struct ElementColours {
    std::vector<int> colours;
    std::vector<int> counts;
    int max = 0;
};

ElementColours colour_elements(const Chunks& chunks, const References& references);

} // namespace op::plan
