#pragma once

// A strategy's kernels on the device: one compiled with the application, its
// launches made by op2_kernels, and one built at run time (op2/rt/gpu/jit.h),
// each launched the same way; and the device symbols the offline kernels read
// their parameters from.  Defined in src/rt/gpu/op_gpu_kernel.cpp.

#include <op2/gpu/shims.h>
#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/jit/params.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::rt::gpu {

// "64 regs", with the local and shared memory when there is any.
std::string describe(const abi::gpu::KernelResources& resources);

// A run-time parameter a built kernel reads from a symbol of its module,
// uploaded when the parameter's value changes.
struct RuntimeSymbol {
    std::size_t param;
    std::string name;
    gpuDrvDevicePtr_t address{};
    // The uploaded value's hash; at first 0, which the zeroed symbol matches.
    std::uint64_t uploaded = 0;
};

// A kernel built at run time from its compiled code, loaded on its first use
// and unloaded when it goes.
class JitKernel {
private:
    std::string m_name;
    std::vector<RuntimeSymbol> m_symbols;
    // The arguments of the READ globals it has baked in, which it never
    // reads from the device.
    std::vector<int> m_baked_globals;
    std::vector<char> m_code;
    bool m_loaded = false;
    int m_max_dynamic_shared_bytes = 0;
    abi::gpu::KernelResources m_resources;
    gpuDrvModule_t m_module{};
    gpuDrvFunction_t m_kernel{};

    void ensure_loaded();

public:
    JitKernel(const JitKernel&) = delete;
    JitKernel(std::vector<char> code, std::string_view name, std::vector<RuntimeSymbol> symbols,
              std::vector<int> baked_globals)
        : m_name{name}, m_symbols{std::move(symbols)},
          m_baked_globals{std::move(baked_globals)}, m_code{std::move(code)} {}
    ~JitKernel();

    const std::vector<int>& baked_globals() const { return m_baked_globals; }

    const abi::gpu::KernelResources& resources();

    // Give its module's symbols the call's values of params, before the
    // call's launches.
    void upload(const op::rt::jit::ParamSet& params);

    void launch(int num_blocks, int block_size, void **args, int shared_bytes);
};

// A strategy's kernel compiled with the application, if any, its attributes
// read and its launches made by op2_kernels.
class OfflineKernel {
private:
    const void *m_function;
    const abi::gpu::OfflineCalls *m_calls;
    abi::gpu::KernelResources m_resources;
    int m_max_dynamic_shared_bytes = 0;

public:
    OfflineKernel(const void *function, const abi::gpu::OfflineCalls *calls);

    // Whether it exists and can launch block_size threads.
    bool usable(int block_size) const {
        return m_function != nullptr && block_size <= m_resources.max_threads;
    }

    const abi::gpu::KernelResources& resources() const { return m_resources; }

    void launch(int num_blocks, int block_size, void **args, int shared_bytes);
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

} // namespace op::rt::gpu
