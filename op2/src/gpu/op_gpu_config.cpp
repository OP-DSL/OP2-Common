#include <gpu/config.h>

#include <gpu/check.h>
#include <gpu/shims.h>
#include <loop/jit.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <string>

namespace op::gpu {

namespace {

Config run_config;

} // namespace

const Config& config() { return run_config; }

void init() {
    namespace env = op::loop::env;
    auto& config = run_config;
    config = {};

    // NVRTC compiles in parallel from CUDA 12.3; earlier, and hipRTC, still
    // gain from a few at once.
#if defined(OP2_CUDA) && CUDART_VERSION >= 12030
    config.build_threads = 16;
#endif

    env::parse_number("OP_CHUNK_COLOURING_WINDOW", config.window);

    int device;
    GPU_CHECK(gpuGetDevice(&device));
    gpuDeviceProp_t props;
    GPU_CHECK(gpuGetDeviceProperties(&props, device));

    std::size_t shared_bytes = props.sharedMemPerBlock;
#ifdef OP2_CUDA
    shared_bytes = std::max(shared_bytes,
                            static_cast<std::size_t>(props.sharedMemPerBlockOptin));
#endif
    // A staged kernel's own static shared memory is a few words.
    constexpr std::size_t static_bytes = 64;
    config.shared_bytes = shared_bytes > static_bytes ? shared_bytes - static_bytes : 0;
    config.multiprocessors = props.multiProcessorCount;
    config.threads_per_multiprocessor = props.maxThreadsPerMultiProcessor;

#ifdef OP2_CUDA
    config.arch = "-arch=sm_" + std::to_string(props.major * 10 + props.minor);
    if (op::loop::jit_policy().debug)
        op_printf("JIT arch flag: %s\n", config.arch.c_str());
#endif
}

} // namespace op::gpu
