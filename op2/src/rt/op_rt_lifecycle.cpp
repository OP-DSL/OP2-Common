#include <op2/rt/lifecycle.h>

#include <op2/rt/loop/loop.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/plan/options.h>

namespace op::rt {

void init() {
    loop::init();
    plan::init_chunk_options();
}

void write_reports(int rank, int ranks) { loop::write_reports(rank, ranks); }

void finalise(int rank, int ranks) { loop::finalise(rank, ranks); }

} // namespace op::rt
