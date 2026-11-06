#include <op2/rt/gpu/kernel.h>

#include <op2/gpu/check.h>

#include <cassert>

namespace op::rt::gpu {

std::string describe(const abi::gpu::KernelResources& resources) {
    std::string out = std::to_string(resources.registers) + " regs";
    if (resources.local_bytes > 0)
        out += ", " + std::to_string(resources.local_bytes) + " B local";
    if (resources.shared_bytes > 0)
        out += ", " + std::to_string(resources.shared_bytes) + " B shared";
    return out;
}

// --- Run-time kernels ----------------------------------------------------------------

JitKernel::~JitKernel() {
    if (m_loaded)
        GPU_DRV_CHECK(gpuDrvModuleUnload(m_module));
}

void JitKernel::ensure_loaded() {
    if (m_loaded)
        return;
    GPU_DRV_CHECK(gpuDrvModuleLoadData(&m_module, m_code.data()));
    GPU_DRV_CHECK(gpuDrvModuleGetFunction(&m_kernel, m_module, m_name.c_str()));
    auto attribute = [&](gpuDrvFuncAttribute_t which) {
        int value = 0;
        GPU_DRV_CHECK(gpuDrvFuncGetAttribute(&value, which, m_kernel));
        return value;
    };
    m_resources = {.registers = attribute(gpuDrvFuncAttributeNumRegs),
                   .local_bytes = attribute(gpuDrvFuncAttributeLocalSizeBytes),
                   .shared_bytes = attribute(gpuDrvFuncAttributeSharedSizeBytes),
                   .max_threads = attribute(gpuDrvFuncAttributeMaxThreadsPerBlock)};
    for (auto& symbol : m_symbols) {
        std::size_t bytes = 0;
        GPU_DRV_CHECK(
            gpuDrvModuleGetGlobal(&symbol.address, &bytes, m_module, symbol.name.c_str()));
    }
    m_loaded = true;
    m_code = {};
}

const abi::gpu::KernelResources& JitKernel::resources() {
    ensure_loaded();
    return m_resources;
}

void JitKernel::upload(const op::rt::jit::ParamSet& params) {
    ensure_loaded();
    for (auto& symbol : m_symbols) {
        const auto& param = params[symbol.param];
        if (param.hash() == symbol.uploaded)
            continue;
        GPU_DRV_CHECK(gpuDrvMemcpyHtoDAsync(symbol.address, param.data(), param.bytes(), 0));
        symbol.uploaded = param.hash();
    }
}

void JitKernel::launch(int num_blocks, int block_size, void **args, int shared_bytes) {
    ensure_loaded();
    if (shared_bytes > m_max_dynamic_shared_bytes) {
        GPU_DRV_CHECK(gpuDrvFuncSetAttribute(
            m_kernel, gpuDrvFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        m_max_dynamic_shared_bytes = shared_bytes;
    }
    GPU_DRV_CHECK(gpuDrvLaunchKernel(m_kernel, num_blocks, 1, 1, block_size, 1, 1,
                                     shared_bytes, nullptr, args, nullptr));
    GPU_CHECK(gpuPeekAtLastError());
}

// --- Offline kernels -----------------------------------------------------------------

OfflineKernel::OfflineKernel(const void *function, const abi::gpu::OfflineCalls *calls)
    : m_function{function}, m_calls{calls} {
    if (function != nullptr)
        m_resources = calls->resources(function);
}

void OfflineKernel::launch(int num_blocks, int block_size, void **args, int shared_bytes) {
    assert(m_function != nullptr);
    if (shared_bytes > m_max_dynamic_shared_bytes) {
        m_calls->allow_shared(m_function, shared_bytes);
        m_max_dynamic_shared_bytes = shared_bytes;
    }
    m_calls->launch(m_function, num_blocks, block_size, args, shared_bytes);
}

// --- Symbols -------------------------------------------------------------------------

void SymbolBindings::upload(const op::rt::jit::ParamSet& params) {
    for (auto& binding : m_bindings) {
        const auto& param = params[binding.param];
        auto& uploaded = binding.shared_hash != nullptr ? *binding.shared_hash : binding.hash;
        auto hash = param.hash();
        if (hash == uploaded)
            continue;
        GPU_CHECK(gpuMemcpyAsync(binding.data_d, param.data(), param.bytes(),
                                 gpuMemcpyHostToDevice));
        uploaded = hash;
    }
}

} // namespace op::rt::gpu
