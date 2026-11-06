#include <op2/rt/gpu/heuristics.h>

#include <op2/env.h>
#include <op2/rt/abi/gpu/launch.h>

namespace op::rt::gpu {

const Defaults& defaults() {
#ifdef OP2_HIP
    // Shared and global fp64 atomics are native: singles unstaged.
    static const Defaults hip{.block_sizes = {abi::gpu::default_block_size, 256},
                              .unstaged_singles = {1, 0},
                              .wavefront = {0},
                              .chunk_atomics = true,
                              .chunk_colouring = false};
    return hip;
#else
    // Shared fp64 adds are compare-and-swap loops: singles stay staged.
    static const Defaults cuda{.block_sizes = {abi::gpu::default_block_size, 64, 256},
                               .unstaged_singles = {0},
                               .wavefront = {0},
                               .chunk_atomics = false,
                               .chunk_colouring = false};
    return cuda;
#endif
}

HeuristicOptions read_heuristic_options() {
    HeuristicOptions options;
    env::parse_number("OP_JIT_MIN_REGISTERS", options.min_registers);
    return options;
}

loop::Estimate estimate(const HeuristicOptions& options, const KernelFacts& facts,
                        loop::KernelKind kind) {
    if (kind == loop::KernelKind::offline) {
        if (!facts.offline)
            return {.reason = "kernel_unavailable"};
        return {.available = true, .cost = 1.0f};
    }
    if (!facts.jit)
        return {.reason = "kernel_unavailable"};
    bool prior = facts.resources.registers > options.min_registers;
    return {.available = true, .cost = prior ? 0.9f : 1.0f};
}

} // namespace op::rt::gpu
