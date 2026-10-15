#include <gpu/config.h>

#include <gpu/shims.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

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

    if (const char *text = std::getenv("OP_AUTOTUNE_BLOCK_SIZES")) {
        std::string_view rest = text;
        while (!rest.empty()) {
            auto comma = rest.find(',');
            auto item = rest.substr(0, comma);
            rest = comma == std::string_view::npos ? std::string_view{}
                                                   : rest.substr(comma + 1);
            int size = std::atoi(std::string{item}.c_str());
            if (size > 0)
                config.block_sizes.push_back(size);
            else if (!item.empty())
                std::fprintf(stderr,
                             "warning: ignoring OP_AUTOTUNE_BLOCK_SIZES item '%s'\n",
                             std::string{item}.c_str());
        }
    }

#ifdef OP2_HIP
    // Shared and global fp64 atomics are both native, so a chunk's single
    // references go unstaged; under CUDA a pointer that may be either makes
    // every increment a generic atomic, and shared fp64 adds are
    // compare-and-swap loops.
    config.unstaged_singles = true;
#endif
    if (env::parse_bool("OP_CHUNK_ATOMICS_UNSTAGED_SINGLES", config.unstaged_singles))
        op_printf("%s OP_CHUNK_ATOMICS_UNSTAGED_SINGLES\n",
                  config.unstaged_singles ? "Enabling" : "Disabling");
    if (env::parse_bool("OP_CHUNK_COLOURING_WAVEFRONT", config.wavefront))
        op_printf("%s OP_CHUNK_COLOURING_WAVEFRONT\n",
                  config.wavefront ? "Enabling" : "Disabling");
    env::parse_number("OP_CHUNK_COLOURING_WINDOW", config.window);

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

#ifdef OP2_CUDA
    config.arch = "-arch=sm_" + std::to_string(props.major * 10 + props.minor);
    if (config.debug)
        op_printf("JIT arch flag: %s\n", config.arch.c_str());
#endif
}

} // namespace op::gpu
