#include <op2/rt/gpu/backend.h>

#include <op2/env.h>
#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/context.h>
#include <op2/rt/gpu/strategy.h>
#include <op2/rt/lifecycle.h>
#include <op_lib_core.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

namespace op::rt::gpu {

Backend::Backend() : heuristics{read_heuristic_options()} {
    // NVRTC compiles in parallel from CUDA 12.3; earlier, and hipRTC, still
    // gain from a few at once.
#if defined(OP2_CUDA) && CUDART_VERSION >= 12030
    options.build_threads = 16;
#endif
    env::parse_number("OP_CHUNK_COLOURING_WINDOW", options.window);
    env::parse_number("OP_CHUNK_ATOMICS_MIN_COMPRESSION", options.min_compression);

    int id;
    GPU_CHECK(gpuGetDevice(&id));
    gpuDeviceProp_t props;
    GPU_CHECK(gpuGetDeviceProperties(&props, id));

    std::size_t shared_bytes = props.sharedMemPerBlock;
#ifdef OP2_CUDA
    shared_bytes = std::max(shared_bytes,
                            static_cast<std::size_t>(props.sharedMemPerBlockOptin));
#endif
    // A staged kernel's own static shared memory is a few words.
    constexpr std::size_t static_bytes = 64;
    device.shared_bytes = shared_bytes > static_bytes ? shared_bytes - static_bytes : 0;
    device.multiprocessors = props.multiProcessorCount;
    device.threads_per_multiprocessor = props.maxThreadsPerMultiProcessor;
#ifdef OP2_CUDA
    device.arch = "-arch=sm_" + std::to_string(props.major * 10 + props.minor);
#endif
}

} // namespace op::rt::gpu

namespace op::rt {

void init_gpu() {
    auto& run = context();
    auto backend = std::make_unique<gpu::Backend>();
    if (run.jit.debug && !backend->device.arch.empty())
        op_printf("JIT arch flag: %s\n", backend->device.arch.c_str());
    run.builds.set_limit(run.jit.max_threads.value_or(backend->options.build_threads));
    run.backend = std::move(backend);
    run.families.declare(gpu::strategy_families());
}

} // namespace op::rt
