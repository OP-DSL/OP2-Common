// The GPU heuristics: a specialised kernel is estimated cheaper than an
// offline one using more than OP_JIT_MIN_REGISTERS registers, and a kind the
// strategy lacks is unavailable.

#include <op2/rt/gpu/heuristics.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "../test_support.h"

namespace gpu = op::rt::gpu;
namespace loop = op::rt::loop;

namespace {

// Whether the specialised kernel of an offline kernel of registers
// registers, or of unknown ones when negative, is estimated cheaper.
bool jit_cheaper(int registers) {
    gpu::KernelFacts facts{true, {}, true};
    if (registers >= 0)
        facts.resources = op::rt::abi::gpu::KernelResources{registers, 0, 0, 1024};
    auto offline = gpu::estimate(facts, loop::KernelKind::offline);
    auto jit = gpu::estimate(facts, loop::KernelKind::jit);
    CHECK(offline.available && offline.cost && jit.available && jit.cost);
    return *jit.cost < *offline.cost;
}

void test_register_rule() {
    gpu::init_heuristics();
    CHECK(jit_cheaper(64));
    CHECK(!jit_cheaper(32));
    CHECK(!jit_cheaper(-1));
    setenv("OP_JIT_MIN_REGISTERS", "64", 1);
    gpu::init_heuristics();
    CHECK(!jit_cheaper(64));
    CHECK(jit_cheaper(65));
    unsetenv("OP_JIT_MIN_REGISTERS");
    gpu::init_heuristics();
}

void test_unavailable() {
    auto none = gpu::estimate({false, {}, false}, loop::KernelKind::offline);
    CHECK(!none.available && none.reason == "kernel_unavailable" && !none.cost);
    none = gpu::estimate({false, {}, false}, loop::KernelKind::jit);
    CHECK(!none.available && none.reason == "kernel_unavailable");
}

} // namespace

int main() {
    try {
        test_register_rule();
        test_unavailable();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return EXIT_FAILURE;
    }

    std::printf("heuristics tests passed\n");
    return EXIT_SUCCESS;
}
