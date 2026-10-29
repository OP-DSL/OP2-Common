#include <op2/rt/plan/options.h>

#include <op2/env.h>

namespace op::rt::plan {

namespace {

ChunkOptions run_options;

} // namespace

const ChunkOptions& chunk_options() { return run_options; }

void init_chunk_options() {
    run_options = {};
    op::env::parse_number("OP_CHUNK_ATOMICS_MIN_COMPRESSION",
                                run_options.min_compression);
}

} // namespace op::rt::plan
