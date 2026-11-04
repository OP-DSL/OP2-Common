#pragma once

// A loop's parts as the runtime makes them from a descriptor in a context:
// its parameters, its backend's executor and the ladder of its strategies,
// for tests that prepare and select its candidates themselves.  The
// descriptor and the context must outlive them.

#include <op2/rt/context.h>
#include <op2/rt/jit/params.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/ladder.h>
#include <op2/rt/loop/strategy.h>
#include <op_profile.h>

template<typename Executor>
struct LoopParts {
    const char *name;
    op::rt::jit::ParamSet params;
    Executor executor;
    op::rt::loop::Ladder ladder;

    template<typename Descriptor>
    LoopParts(Descriptor& descriptor, op::rt::Context& context)
        : name{descriptor.name}, params{descriptor.name, context.jit},
          executor{descriptor, params, context},
          ladder{descriptor.name, false, context.families, context.jit} {
        executor.add_strategies(ladder);
    }

    // One call as a loop runs it once its halos are exchanged, in the
    // loop's profile scopes, which a strategy's phases switch between: the
    // parameters take its values, the ladder selects, and the selection runs.
    op::rt::loop::Prepared run(const op::rt::loop::Call& call) {
        params.update(call.args);
        auto selection = ladder.select(call);
        auto& strategy = *selection.candidate->strategy;
        op_profile_enter_kernel(name, "test", "test");
        op_profile_enter("Kernel");
        strategy.run(selection, call, false);
        op_profile_exit();
        op_profile_exit();
        return selection;
    }
};
