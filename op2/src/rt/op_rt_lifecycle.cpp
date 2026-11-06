#include <op2/rt/lifecycle.h>

#include <op2/rt/context.h>

#include <cassert>

namespace op::rt {

namespace {

// Destroyed by finalise or a later init, never at exit: a program that skips
// op_exit must not free device state after the driver has shut down.
Context *current = nullptr;

} // namespace

Context& context() {
    assert(current != nullptr);
    return *current;
}

void init() {
    delete current;
    current = new Context;
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
