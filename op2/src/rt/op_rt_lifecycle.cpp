#include <op2/rt/lifecycle.h>

#include <op2/rt/jit/options.h>
#include <op2/rt/loop/family.h>
#include <op2/rt/loop/loop.h>
#include <op2/rt/plan/options.h>
#include <op2/rt/tune/autotune.h>
#include <op2/rt/tune/heuristics.h>

namespace op::rt {

void init() {
    loop::init();
    jit::init();
    tune::init_autotune();
    tune::init_heuristics();
    plan::init_chunk_options();
}

void write_reports(int rank, int ranks) { loop::write_reports(rank, ranks); }

void finalise(int rank, int ranks) { loop::finalise(rank, ranks); }

} // namespace op::rt
