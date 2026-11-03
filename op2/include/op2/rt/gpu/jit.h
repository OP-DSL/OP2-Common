#pragma once

// The CUDA and HIP JIT mechanism: the device symbols the offline kernels read
// their parameters from, the kernels built through NVRTC or hipRTC, how those
// have their run-time parameters, and their launches.  The policy, the
// parameters and the build cache are op2/rt/jit's.  Defined in
// src/rt/gpu/op_gpu_jit.cpp.

#include <op2/gpu/shims.h>
#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/jit/cache.h>
#include <op2/rt/jit/params.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::gpu {

// --- Kernels --------------------------------------------------------------------------

// "64 regs", with the local and shared memory when there is any.
std::string describe(const op::rt::abi::gpu::KernelResources& resources);

// A run-time parameter a built kernel reads from a symbol of its module,
// uploaded when the parameter's value changes.
struct RuntimeSymbol {
    std::size_t param;
    std::string name;
    gpuDrvDevicePtr_t address{};
    // The uploaded value's hash; at first 0, which the zeroed symbol matches.
    std::uint64_t uploaded = 0;
};

// A kernel built at run time, loaded on its first launch or when its
// resources are read, and unloaded when it goes.
class JitKernel {
private:
    std::string m_name;
    std::vector<RuntimeSymbol> m_symbols;
    // The arguments of the READ globals it has baked in, which it never
    // reads from the device.
    std::vector<int> m_baked_globals;
    bool m_loaded = false;
    int m_max_dynamic_shared_bytes = 0;
    op::rt::abi::gpu::KernelResources m_resources;
    char *m_cubin;
    gpuDrvModule_t m_module;
    gpuDrvFunction_t m_kernel;

    void ensure_loaded();

public:
    JitKernel(const JitKernel&) = delete;
    JitKernel(char *cubin, std::string_view name, std::vector<RuntimeSymbol> symbols,
              std::vector<int> baked_globals)
        : m_name{name}, m_symbols{std::move(symbols)},
          m_baked_globals{std::move(baked_globals)}, m_cubin{cubin} {}
    ~JitKernel();

    const std::vector<int>& baked_globals() const { return m_baked_globals; }

    const op::rt::abi::gpu::KernelResources& resources();

    // Launch, its module's symbols first given the call's values of params.
    void launch(int num_blocks, int block_size, void **args, int shared_bytes,
                const op::rt::jit::ParamSet& params);
};

// A strategy's kernel compiled with the application, if any, its attributes
// read and its launches made by op2_kernels.
class OfflineKernel {
private:
    const void *m_function;
    const abi::gpu::OfflineCalls *m_calls;
    op::rt::abi::gpu::KernelResources m_resources;
    int m_max_dynamic_shared_bytes = 0;

public:
    OfflineKernel(const void *function, const abi::gpu::OfflineCalls *calls);

    // Whether it exists and can launch block_size threads.
    bool usable(int block_size) const {
        return m_function != nullptr && block_size <= m_resources.max_threads;
    }

    const op::rt::abi::gpu::KernelResources& resources() const { return m_resources; }

    void launch(int num_blocks, int block_size, void **args, int shared_bytes);
};

// --- Parameters ---------------------------------------------------------------------

// What a strategy fixes for its kernel: the launch, and its variant's
// switches.  Passed to each acquire and upload, never kept between them.
struct LaunchValues {
    int global_stride = 0;
    int block_size = 0;
    bool unstaged_singles = false;
    bool wavefront = false;

    // The values by slot (abi::gpu::LaunchValue).
    std::array<int, abi::gpu::launch_values> slots() const {
        using abi::gpu::LaunchValue;
        std::array<int, abi::gpu::launch_values> out{};
        out[static_cast<std::size_t>(LaunchValue::global_stride)] = global_stride;
        out[static_cast<std::size_t>(LaunchValue::block_size)] = block_size;
        out[static_cast<std::size_t>(LaunchValue::unstaged_singles)] = unstaged_singles;
        out[static_cast<std::size_t>(LaunchValue::wavefront)] = wavefront;
        return out;
    }
};

// The device symbols the offline kernels read parameters from, each
// uploaded when its value has changed since.
class SymbolBindings {
private:
    struct Binding {
        std::size_t param;
        void *data_d;
        // The uploaded value's hash: shared by every loop binding the symbol
        // when given, else hash; at first 0, which the zeroed symbol matches.
        std::uint64_t *shared_hash;
        std::uint64_t hash = 0;
    };

    std::vector<Binding> m_bindings;

public:
    void bind(std::size_t param, void *data_d, std::uint64_t *shared_hash) {
        m_bindings.push_back({param, data_d, shared_hash});
    }

    void upload(const op::rt::jit::ParamSet& params);
};

// --- Builds -------------------------------------------------------------------------

// Builds a loop's specialised kernels through NVRTC or hipRTC.
class Compiler {
private:
    std::string m_loop;
    abi::gpu::JitHeaders m_headers;

public:
    // A build of one kernel, which returns it, or none when the compile
    // failed after printing the compiler's log.  It holds what it reads, so
    // its loop's compiler may go before the build is joined.
    struct Job {
        std::string loop;
        abi::gpu::JitHeaders headers;
        std::string text;
        std::string name;
        std::vector<RuntimeSymbol> symbols;
        std::vector<int> baked_globals;

        std::unique_ptr<JitKernel> operator()();
    };

    Compiler(std::string_view loop, abi::gpu::JitHeaders headers)
        : m_loop{loop}, m_headers{std::move(headers)} {}

    // The job that builds family's kernel with the parameters' current values
    // and the strategy's, by slot, baked in, or at run time as their modes say.
    Job job(const op::rt::jit::ParamSet& params, std::span<const int> values,
            const abi::gpu::Family& family) const;
};

} // namespace op::rt::gpu
