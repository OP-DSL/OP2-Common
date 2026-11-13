#include <op2/rt/lifecycle.h>

#include <op2/rt/context.h>

#include <cassert>

namespace op::rt {

namespace {

// Destroyed by finalise or a later init, never at exit: a program that skips
// op_exit must not free device state after the driver has shut down.
Context *current = nullptr;

// Stops the context's builds when the thread that made it leaves without
// op_exit.  exit() runs thread_local destructors before its handlers, among
// them the teardown a JIT compiler registers on its first build (NVRTC's),
// which a build under way crashes in.
struct StopBuilds {
    ~StopBuilds() {
        if (current != nullptr)
            current->builds.stop();
    }
};
thread_local StopBuilds stop_builds;

} // namespace

Context& context() {
    assert(current != nullptr);
    return *current;
}

void init() {
    delete current;
    current = new Context;
    // Made on its first use in this thread.
    static_cast<void>(&stop_builds);
}

void write_reports(int rank, int ranks) {
    if (current != nullptr)
        current->loops.write_reports(*current, rank, ranks, false);
}

void finalise(int rank, int ranks) {
    if (current == nullptr)
        return;
    current->loops.write_reports(*current, rank, ranks, true);
    delete current;
    current = nullptr;
}

} // namespace op::rt
