#include <gpu/config.h>

#include <gpu/shims.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace op::gpu {

namespace {

Config run_config;

void check(gpuError_t result, const char *call) {
    if (result == gpuSuccess)
        return;
    std::fprintf(stderr, "error: %s failed with %s\n", call, gpuGetErrorString(result));
    std::exit(1);
}

} // namespace

const Config& config() { return run_config; }

void init() {
    namespace env = op::loop::env;
    auto& config = run_config;
    config = {};

#if defined(OP2_CUDA) && CUDART_VERSION >= 12030
    config.max_threads = 16;
#endif
    env::parse_bool("OP_JIT_ENABLE", config.enable);
    if (!config.enable)
        op_printf("Disabling JIT compilation\n");
    env::parse_bool("OP_JIT_DEBUG", config.debug);
    if (config.debug)
        op_printf("Enabling JIT debug\n");
    env::parse_bool("OP_JIT_SEQ_COMPILE", config.seq_compile);
    env::parse_bool("OP_JIT_FORCE", config.force);
    if (config.force)
        op_printf("Forcing JIT compilation regardless of register count\n");
    env::parse_number("OP_JIT_MAX_THREADS", config.max_threads);
    env::parse_number("OP_JIT_MIN_CALLS", config.min_calls);
    env::parse_number("OP_JIT_MIN_REGISTERS", config.min_registers);

    env::parse_number("OP_CHUNK_COLOURING_WINDOW", config.window);

    env::parse_bool("OP_LAZY", config.lazy);
    env::parse_number("OP_LAZY_MAX_CALLS", config.lazy_max_calls);
    env::parse_number("OP_LAZY_COPY_BYTES", config.lazy_copy_bytes);
    config.lazy_max_calls = std::max(config.lazy_max_calls, 1);
    env::parse_bool("OP_FUSION", config.fusion);
    env::parse_number("OP_FUSION_MAX_REGISTERS", config.fusion_max_registers);
    env::parse_bool("OP_FUSION_TUNE", config.fusion_tune);
    env::parse_bool("OP_FUSION_OVERLAP", config.fusion_overlap);
    if (const char *dump = std::getenv("OP_FUSION_DUMP"))
        config.fusion_dump = dump;

    int device;
    check(gpuGetDevice(&device), "gpuGetDevice");
    gpuDeviceProp_t props;
    check(gpuGetDeviceProperties(&props, device), "gpuGetDeviceProperties");

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
    config.registers_per_multiprocessor = props.regsPerMultiprocessor;
    config.warp_size = props.warpSize;
#ifdef OP2_CUDA
    config.blocks_per_multiprocessor = props.maxBlocksPerMultiProcessor;
#else
    config.blocks_per_multiprocessor = 32;
#endif

#ifdef OP2_CUDA
    config.arch = "-arch=sm_" + std::to_string(props.major * 10 + props.minor);
    if (config.debug)
        op_printf("JIT arch flag: %s\n", config.arch.c_str());
#endif
}

} // namespace op::gpu
