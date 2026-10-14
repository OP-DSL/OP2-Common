#pragma once

// Partition a set into chunks that each get private scratch for the targets
// they update: the plans of the chunked strategies.  Host code with no
// device knowledge; a strategy supplies what its chunks can hold and encodes
// the partition for its kernels.

#include <loop/chunk_layout.h>
#include <loop/strategy.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace op::chunk {

// Run-wide options, read once by init_options.
struct Options {
    // OP_CHUNK_ATOMICS_MIN_COMPRESSION: staged partitions averaging fewer
    // references per flushed target than this are rejected; 0 accepts all.
    double min_compression = 1.5;
    // OP_CHUNK_ATOMICS_CHUNK_SIZE, OP_CHUNK_COLOURING_CHUNK_SIZE: elements
    // per chunk; zero leaves it to OP_part_size and the lanes.
    int atomics_chunk_size = 0;
    int colouring_chunk_size = 0;
    // OP_CHUNK_COLOURING_WINDOW: chunks per window of a wavefront partition;
    // zero leaves it to the planner.
    int window = 0;
    // Variant bits: OP_CHUNK_ATOMICS_UNSTAGED_SINGLES and
    // OP_CHUNK_COLOURING_WAVEFRONT over the backend's defaults.
    int variants = 0;
};

const Options& options();
void init_options(int default_variants);

// The indirect arguments a chunked strategy stages, grouped by dat as its
// kernel lays its regions out: chunk_atomics stages every increment,
// chunk_colouring every argument on a dat the loop increments or
// read-writes.  Generated with the kernel it describes.
struct ArgGroup {
    int arg;
    int group;
};

struct ArgGroups {
    std::span<const ArgGroup> args;
    int groups = 0;
};

// What a strategy's chunks can hold.
struct Constraints {
    // Workers that process a chunk together; chunks are multiples of it.
    int lanes = 128;
    // Scratch a chunk may use.
    std::size_t scratch_bytes = 0;
    // Slots a region may hold.
    int max_slots = chunk::max_slots;
    // Requested chunk size; zero takes the options' or OP_part_size.
    int elements = 0;
};

struct Statistics {
    std::size_t raw_references = 0;
    std::size_t distinct_targets = 0;
    // References left unstaged, the only one their chunk makes to their
    // target; counted in raw_references and distinct_targets.
    std::size_t unstaged_singles = 0;
    // Batches over all sections, and the most thread colours any chunk needs.
    std::size_t batches = 0;
    int max_thread_colours = 0;

    // The staged references per global flush.
    double compression() const {
        return distinct_targets > 0
                   ? static_cast<double>(raw_references) /
                         static_cast<double>(distinct_targets)
                   : 0.0;
    }
};

// Scratch slots for every staged reference.  A region is one chunk's slots of
// one staged dat: stage_counts has its slot count and slot_offsets where
// slot_targets lists its targets, in ascending order.
struct Staging {
    std::vector<StageWord> stage_words;
    std::vector<int> stage_counts;
    std::vector<int> slot_offsets;
    std::vector<int> slot_targets;
    // Per chunk, the scratch bytes its regions need.
    std::vector<std::size_t> chunk_scratch_bytes;
};

// Chunk and thread colours.  Each section runs one batch per chunk colour;
// chunk_order lists the chunks batch by batch, as the batch offsets index
// them.
//
// A wavefront partition runs each section as one batch instead: chunk_order
// lists its chunks window by window, chunk colour by chunk colour, and a
// worker takes the next position once that position's predecessors, the
// latest earlier position sharing each of its targets, are done.
struct Colouring {
    std::vector<int> chunk_order;
    std::vector<std::uint8_t> thread_colours;
    std::vector<int> chunk_thread_colours;

    bool wavefront = false;
    // Per chunk_order position, CSR over predecessor positions.
    std::vector<int> wave_pred_offsets;
    std::vector<int> wave_preds;
};

// Consecutive source chunks inside each section, their staging, and the
// batches of chunks that may run concurrently.
struct Partition {
    int chunk_size = 0;
    int set_stride = 0;

    std::vector<int> source_offsets;
    std::vector<int> section_chunk_offsets;

    Staging staging;
    std::optional<Colouring> colouring;

    // Per section its first batch, per batch its first chunk (a chunk_order
    // position under chunk colouring), and the scratch its largest chunk
    // needs.  Without chunk colouring a section is one batch.
    std::vector<int> section_batch_offsets;
    std::vector<int> batch_chunk_offsets;
    std::vector<std::size_t> batch_scratch_bytes;

    Statistics statistics;

    std::size_t num_chunks() const {
        return source_offsets.empty() ? 0 : source_offsets.size() - 1;
    }

    // The chunk a chunk_order position names, or the position itself
    // without chunk colouring.
    int chunk_at(int position) const {
        return colouring ? colouring->chunk_order[static_cast<std::size_t>(position)]
                         : position;
    }

    std::size_t max_chunk_scratch_bytes(int begin, int end) const {
        std::size_t bytes = 0;
        for (int i = begin; i < end; ++i)
            bytes = std::max(bytes, staging.chunk_scratch_bytes[
                                        static_cast<std::size_t>(chunk_at(i))]);
        return bytes;
    }

    int batches(int section) const {
        auto index = static_cast<std::size_t>(section);
        return section_batch_offsets[index + 1] - section_batch_offsets[index];
    }

    int batch_index(int section, int batch) const {
        return section_batch_offsets[static_cast<std::size_t>(section)] + batch;
    }

    // The positions one batch covers.
    std::pair<int, int> batch_range(int section, int batch) const {
        auto index = static_cast<std::size_t>(batch_index(section, batch));
        return {batch_chunk_offsets[index], batch_chunk_offsets[index + 1]};
    }

    std::size_t scratch_bytes(int section, int batch) const {
        return batch_scratch_bytes[
            static_cast<std::size_t>(batch_index(section, batch))];
    }
};

enum class Kind {
    staged,    // concurrent chunks may share targets: every slot flushes atomically
    coloured,  // chunks of a batch share no target: slots are seeded and stored
};

// rejected names the reason when the configuration does not partition
// acceptably: no_active_argument, incompatible_argument,
// insufficient_scratch, low_compression, too_many_colours.
struct Result {
    std::string_view rejected;
    std::unique_ptr<Partition> partition;

    explicit operator bool() const {
        return rejected.empty() && partition != nullptr;
    }
};

Result build_partition(Kind kind, const op::loop::Configuration& configuration,
                       const op::loop::Sections& sections,
                       const ArgGroups& groups, const Constraints& constraints,
                       const Options& options);

inline Result build_partition(Kind kind,
                              const op::loop::Configuration& configuration,
                              const op::loop::Sections& sections,
                              const ArgGroups& groups,
                              const Constraints& constraints) {
    return build_partition(kind, configuration, sections, groups, constraints,
                           options());
}

} // namespace op::chunk
