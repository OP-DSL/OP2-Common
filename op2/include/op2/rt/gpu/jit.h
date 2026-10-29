#pragma once

// The CUDA and HIP JIT mechanism: the device symbols the offline kernels read
// their parameters from, the kernels built through NVRTC or hipRTC, how those
// have their run-time parameters, and their launches.  The policy, the
// parameters and the build cache are op2/rt/loop/jit.h's.  Defined in
// src/rt/gpu/op_gpu_jit.cpp.

#include <op2/gpu/shims.h>
#include <op2/rt/gpu/loop.h>
#include <op2/rt/loop/jit.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::gpu {

// --- Kernels --------------------------------------------------------------------------

// A run-time parameter a built kernel reads from a symbol of its module,
// uploaded when the parameter's value changes.
struct RuntimeSymbol {
    std::size_t param;
    std::string name;
    gpuDrvDevicePtr_t address{};
    // The uploaded value's hash; at first 0, which the zeroed symbol matches.
    std::uint64_t uploaded = 0;
};

// A kernel built at run time, loaded on its first launch or when its figures
// are read.
class JitKernel {
private:
    std::string m_name;
    std::vector<RuntimeSymbol> m_symbols;
    // The arguments of the READ globals it has baked in, which it never
    // reads from the device.
    std::vector<int> m_baked_globals;
    bool m_loaded = false;
    int m_max_dynamic_shared_bytes = 0;
    KernelFigures m_figures;
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
    ~JitKernel() { delete[] m_cubin; }

    const std::vector<int>& baked_globals() const { return m_baked_globals; }

    const KernelFigures& figures();

    // Launch, its module's symbols first given the call's values of params.
    void launch(int num_blocks, int block_size, void **args, int shared_bytes,
                const op::rt::loop::ParamSet& params);
};

// A strategy's kernel compiled with the application, if any, its attributes
// read and its launches made by op2_kernels.
class OfflineKernel {
private:
    OfflineFunction m_function;
    KernelFigures m_figures;
    int m_max_dynamic_shared_bytes = 0;

public:
    explicit OfflineKernel(OfflineFunction function);

    // Whether it exists and can launch block_size threads.
    bool usable(int block_size) const {
        return m_function.function != nullptr && block_size <= m_figures.max_threads;
    }

    int registers() const { return m_figures.registers; }
    const KernelFigures& figures() const { return m_figures; }

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

    // The values by slot.
    std::array<int, 4> slots() const {
        return {global_stride, block_size, unstaged_singles, wavefront};
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

    void upload(op::rt::loop::ParamSet& params);
};

// --- Builds -------------------------------------------------------------------------

// Builds a loop's specialised kernels through NVRTC or hipRTC.
class Compiler {
private:
    std::string m_loop;
    JitHeaders m_headers;

public:
    // A build of one kernel, which returns it, or none when the compile
    // failed after printing the compiler's log.
    struct Job {
        const Compiler *compiler;
        std::string text;
        std::string name;
        std::vector<RuntimeSymbol> symbols;
        std::vector<int> baked_globals;

        std::unique_ptr<JitKernel> operator()();
    };

    Compiler(std::string_view loop, JitHeaders headers)
        : m_loop{loop}, m_headers{std::move(headers)} {}

    // The job that builds source's kernel with the parameters' current
    // values baked in, or at run time as their modes say.
    Job job(const op::rt::loop::ParamSet& params, const KernelSource& source) const;
};

} // namespace op::rt::gpu
