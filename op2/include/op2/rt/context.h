#pragma once

// The runtime from op_init to op_exit: the run-wide options, the families
// the backend declared, the threads specialised kernels build on, what the
// backend keeps for its loops, and the loops made so far.  op::rt::init
// makes it and the backend's init adds its part (op2/rt/lifecycle.h);
// finalise destroys it, its loops first.  Defined in
// src/rt/op_rt_lifecycle.cpp.

#include <op2/rt/jit/options.h>
#include <op2/rt/jit/pool.h>
#include <op2/rt/loop/family.h>
#include <op2/rt/loop/registry.h>
#include <op2/rt/tune/options.h>

#include <memory>

namespace op::rt {

// What a backend keeps for the run beyond its loops: its options, its
// device's facts and the plans its loops share.
class Backend {
public:
    virtual ~Backend() = default;
};

struct Context {
    jit::Options jit = jit::read_options();
    tune::AutotuneOptions autotune = tune::read_options();
    loop::Families families;
    // Limited by the backend, which builds on it.
    jit::BuildPool builds;
    std::unique_ptr<Backend> backend;
    // Last, so the loops go before what they use.
    loop::Registry loops;
};

// The run's context, which op_init made: the generated code's way into the
// runtime.
Context& context();

} // namespace op::rt
