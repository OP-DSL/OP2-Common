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

// Whether, under options, the specialised kernel of an offline kernel of
// registers registers, none known when 0, is estimated cheaper.
bool jit_cheaper(const gpu::HeuristicOptions& options, int registers) {
    gpu::KernelFacts facts{.offline = true,
                           .resources = {.registers = registers, .max_threads = 1024},
                           .jit = true};
    auto offline = gpu::estimate(options, facts, loop::KernelKind::offline);
    auto jit = gpu::estimate(options, facts, loop::KernelKind::jit);
    CHECK(offline.available && offline.cost && jit.available && jit.cost);
    return *jit.cost < *offline.cost;
}

void test_register_rule() {
    auto options = gpu::read_heuristic_options();
    CHECK(jit_cheaper(options, 64));
    CHECK(!jit_cheaper(options, 32));
    CHECK(!jit_cheaper(options, 0));
    setenv("OP_JIT_MIN_REGISTERS", "64", 1);
    options = gpu::read_heuristic_options();
    CHECK(!jit_cheaper(options, 64));
    CHECK(jit_cheaper(options, 65));
    unsetenv("OP_JIT_MIN_REGISTERS");
}

void test_unavailable() {
    const gpu::HeuristicOptions options;
    auto none = gpu::estimate(options, {}, loop::KernelKind::offline);
    CHECK(!none.available && none.reason == "kernel_unavailable" && !none.cost);
    none = gpu::estimate(options, {}, loop::KernelKind::jit);
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
