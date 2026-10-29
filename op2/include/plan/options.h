#pragma once

// The chunked strategies' run-wide options, read by init_chunk_options,
// which op_init calls.

namespace op::plan {

struct ChunkOptions {
    // OP_CHUNK_ATOMICS_MIN_COMPRESSION: staged partitions averaging fewer
    // references per flushed target than this are rejected; 0 accepts all.
    double min_compression = 1.5;
};

const ChunkOptions& chunk_options();
void init_chunk_options();

} // namespace op::plan
