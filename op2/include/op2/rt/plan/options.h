#pragma once

// The chunked strategies' run-wide options, which op::rt::init reads into
// the run's context.

namespace op::rt::plan {

struct ChunkOptions {
    // OP_CHUNK_ATOMICS_MIN_COMPRESSION: staged partitions averaging fewer
    // references per flushed target than this are rejected; 0 accepts all.
    double min_compression = 1.5;
};

ChunkOptions read_chunk_options();

} // namespace op::rt::plan
