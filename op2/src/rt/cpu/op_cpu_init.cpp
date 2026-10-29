#include <op2/rt/lifecycle.h>

#include <op2/rt/cpu/strategy.h>
#include <op2/rt/loop/strategy.h>

namespace op::rt::cpu {

// The families in ladder order.  An atomic increment is a compare-and-swap
// loop on most CPUs, so chunk colouring runs over the set colouring floor by
// default.
void init() {
    const op::rt::loop::Family families[] = {
        {"plain", Simple::axes()},
        {"chunk_atomics", ChunkAtomics::axes(), false},
        {"atomics", Simple::axes(), false},
        {"chunk_colouring", ChunkColouring::axes()},
        {"colouring", Colouring::axes()},
    };
    op::rt::loop::declare_families(families);
}

} // namespace op::rt::cpu

namespace op::rt {

void init_cpu() { cpu::init(); }

} // namespace op::rt
