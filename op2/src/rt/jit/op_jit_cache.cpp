#include <op2/rt/jit/cache.h>

#include <atomic>

namespace op::rt::jit {

namespace {

std::atomic_int builds_in_flight = 0;

} // namespace

bool take_build_slot(int limit) {
    int count = builds_in_flight.load();
    while (count < limit)
        if (builds_in_flight.compare_exchange_weak(count, count + 1))
            return true;
    return false;
}

void release_build_slot() { --builds_in_flight; }

} // namespace op::rt::jit
