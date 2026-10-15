#pragma once

// The chunked strategies' run-wide options, which every backend shares.
// Read by init_chunk_options, which op_init calls.

namespace op::plan {

struct ChunkOptions {
    // OP_CHUNK_ATOMICS_MIN_COMPRESSION: staged partitions averaging fewer
    // references per flushed target than this are rejected; 0 accepts all.
    double min_compression = 1.5;
    // OP_CHUNK_ATOMICS_CHUNK_SIZE, OP_CHUNK_COLOURING_CHUNK_SIZE: elements
    // per chunk; zero leaves it to the backend.
    int atomics_chunk_size = 0;
    int colouring_chunk_size = 0;
};

const ChunkOptions& chunk_options();
void init_chunk_options();

// Elements per chunk: the translator's size for the loop, else the
// variable's, else the backend's.
inline int chunk_size(int loop_size, int option_size, int backend_size) {
    return loop_size > 0 ? loop_size : option_size > 0 ? option_size : backend_size;
}

} // namespace op::plan
