#include <plan/options.h>

#include <loop/strategy.h>

namespace op::plan {

namespace {

ChunkOptions run_options;

} // namespace

const ChunkOptions& chunk_options() { return run_options; }

void init_chunk_options() {
    run_options = {};
    op::loop::env::parse_number("OP_CHUNK_ATOMICS_MIN_COMPRESSION",
                                run_options.min_compression);
    op::loop::env::parse_number("OP_CHUNK_ATOMICS_CHUNK_SIZE",
                                run_options.atomics_chunk_size);
    op::loop::env::parse_number("OP_CHUNK_COLOURING_CHUNK_SIZE",
                                run_options.colouring_chunk_size);
}

} // namespace op::plan
