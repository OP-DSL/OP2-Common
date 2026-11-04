#include <op2/rt/plan/options.h>

#include <op2/env.h>

namespace op::rt::plan {

ChunkOptions read_chunk_options() {
    ChunkOptions options;
    op::env::parse_number("OP_CHUNK_ATOMICS_MIN_COMPRESSION", options.min_compression);
    return options;
}

} // namespace op::rt::plan
