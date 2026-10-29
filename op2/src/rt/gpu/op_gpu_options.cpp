#include <op2/rt/gpu/options.h>

#include <op2/env.h>
#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op2/rt/gpu/strategy.h>
#include <op2/rt/lifecycle.h>
#include <op2/rt/jit/cache.h>
#include <op2/rt/jit/params.h>
#include <op2/rt/loop/strategy.h>
#include <op_lib_core.h>

#include <string>

namespace op::rt::gpu {

namespace {

Options run_options;
Device run_device;

} // namespace

const Options& options() { return run_options; }
const Device& device() { return run_device; }

void init() {
    namespace env = op::env;
    auto& options = run_options;
    auto& device = run_device;
    options = {};
    device = {};

    // NVRTC compiles in parallel from CUDA 12.3; earlier, and hipRTC, still
    // gain from a few at once.
#if defined(OP2_CUDA) && CUDART_VERSION >= 12030
    options.build_threads = 16;
#endif

    env::parse_number("OP_CHUNK_COLOURING_WINDOW", options.window);

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
    if (op::rt::jit::options().debug)
        op_printf("JIT arch flag: %s\n", device.arch.c_str());
#endif

    // The families the generated code registers, each a strategy class, in
    // ladder order.  chunk_atomics runs by default on AMD, where its unstaged
    // singles are native atomics.
#ifdef OP2_HIP
    const bool chunk_atomics = true;
#else
    const bool chunk_atomics = false;
#endif
    const op::rt::loop::Family families[] = {
        {"plain", Simple::axes()},
        {"chunk_atomics", ChunkAtomics::axes(), chunk_atomics},
        {"atomics", Simple::axes()},
        {"chunk_colouring", ChunkColouring::axes(), false},
        {"colouring", Colouring::axes()},
    };
    op::rt::loop::declare_families(families);
}

} // namespace op::rt::gpu

namespace op::rt {

void init_gpu() { gpu::init(); }

} // namespace op::rt
