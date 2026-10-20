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
}

} // namespace op::plan
