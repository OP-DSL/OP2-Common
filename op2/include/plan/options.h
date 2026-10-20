#pragma once

// The chunked strategies' run-wide options and their chunk size axis, which
// every backend shares.  The options are read by init_chunk_options, which
// op_init calls.

#include <loop/strategy.h>

namespace op::plan {

struct ChunkOptions {
    // OP_CHUNK_ATOMICS_MIN_COMPRESSION: staged partitions averaging fewer
    // references per flushed target than this are rejected; 0 accepts all.
    double min_compression = 1.5;
};

const ChunkOptions& chunk_options();
void init_chunk_options();

// Elements per chunk, OP_<FAMILY>_CHUNK_SIZES else OP_CHUNK_SIZES, by
// default the backend's size.
inline op::loop::Axis chunk_size_axis(int default_size) {
    return {"CHUNK_SIZES", "chunk_size", false, {default_size}};
}

} // namespace op::plan
