#include <op2/rt/gpu/jit.h>

#include <op2/gpu/check.h>
#include <op2/rt/gpu/options.h>
#include <op_lib_core.h>

#include <cassert>
#include <cstdio>
#include <utility>

namespace op::rt::gpu {

std::string describe(const op::rt::abi::KernelFigures& figures) {
    std::string out = std::to_string(figures.registers) + " regs";
    if (figures.local_bytes > 0)
        out += ", " + std::to_string(figures.local_bytes) + " B local";
    if (figures.shared_bytes > 0)
        out += ", " + std::to_string(figures.shared_bytes) + " B shared";
    return out;
}

// --- Kernels --------------------------------------------------------------------------

void JitKernel::ensure_loaded() {
    if (m_loaded)
        return;
    GPU_DRV_CHECK(gpuDrvModuleLoadData(&m_module, m_cubin));
    GPU_DRV_CHECK(gpuDrvModuleGetFunction(&m_kernel, m_module, m_name.c_str()));
    auto attribute = [&](gpuDrvFuncAttribute_t which) {
        int value = 0;
        GPU_DRV_CHECK(gpuDrvFuncGetAttribute(&value, which, m_kernel));
        return value;
    };
    m_figures = {attribute(gpuDrvFuncAttributeNumRegs),
                 attribute(gpuDrvFuncAttributeLocalSizeBytes),
                 attribute(gpuDrvFuncAttributeSharedSizeBytes),
                 attribute(gpuDrvFuncAttributeMaxThreadsPerBlock)};
    for (auto& symbol : m_symbols) {
        std::size_t bytes = 0;
        GPU_DRV_CHECK(
            gpuDrvModuleGetGlobal(&symbol.address, &bytes, m_module, symbol.name.c_str()));
    }
    m_loaded = true;
    delete[] m_cubin;
    m_cubin = nullptr;
}

const op::rt::abi::KernelFigures& JitKernel::figures() {
    ensure_loaded();
    return m_figures;
}

void JitKernel::launch(int num_blocks, int block_size, void **args, int shared_bytes,
                       const op::rt::jit::ParamSet& params) {
    ensure_loaded();
    for (auto& symbol : m_symbols) {
        const auto& param = params[symbol.param];
        if (param.last_hash() == symbol.uploaded)
            continue;
        GPU_DRV_CHECK(gpuDrvMemcpyHtoDAsync(symbol.address, param.data(), param.bytes(), 0));
        symbol.uploaded = param.last_hash();
    }
    if (shared_bytes > m_max_dynamic_shared_bytes) {
        GPU_DRV_CHECK(gpuDrvFuncSetAttribute(
            m_kernel, gpuDrvFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        m_max_dynamic_shared_bytes = shared_bytes;
    }
    GPU_DRV_CHECK(gpuDrvLaunchKernel(m_kernel, num_blocks, 1, 1, block_size, 1, 1,
                                     shared_bytes, NULL, args, 0));
    GPU_CHECK(gpuPeekAtLastError());
    if (op::rt::jit::options().debug)
        GPU_CHECK(gpuStreamSynchronize(0));
}

OfflineKernel::OfflineKernel(OfflineFunction function) : m_function{function} {
    if (function.function != nullptr)
        m_figures = function.calls->figures(function.function);
}

void OfflineKernel::launch(int num_blocks, int block_size, void **args, int shared_bytes) {
    assert(m_function.function != nullptr);
    if (shared_bytes > m_max_dynamic_shared_bytes) {
        m_function.calls->allow_shared(m_function.function, shared_bytes);
        m_max_dynamic_shared_bytes = shared_bytes;
    }
    m_function.calls->launch(m_function.function, num_blocks, block_size, args,
                             shared_bytes);
    if (op::rt::jit::options().debug)
        GPU_CHECK(gpuStreamSynchronize(0));
}

// --- Parameters ---------------------------------------------------------------------

void SymbolBindings::upload(op::rt::jit::ParamSet& params) {
    for (auto& binding : m_bindings) {
        auto& param = params[binding.param];
        auto& uploaded = binding.shared_hash != nullptr ? *binding.shared_hash : binding.hash;
        auto hash = param.hash();
        if (hash == uploaded)
            continue;
        GPU_CHECK(gpuMemcpyAsync(binding.data_d, param.data(), param.bytes(),
                                 gpuMemcpyHostToDevice));
        uploaded = hash;
    }
}

// --- Builds -------------------------------------------------------------------------

namespace {

// The run-time parameters as a built kernel has them: a global as its
// wrapper's argument, which the source reads through
// op::rt::kernel::param_value; any other as a symbol of its module, added to
// symbols.
std::string runtime_declarations(const op::rt::jit::ParamSet& params,
                                 std::vector<RuntimeSymbol>& symbols) {
    std::string out;
    for (std::size_t i = 0; i < params.size(); ++i) {
        const auto& param = params[i];
        if (param.mode() != op::rt::jit::ParamMode::runtime)
            continue;
        if (param.source() == op::rt::abi::ParamSource::scalar_arg ||
            param.source() == op::rt::abi::ParamSource::array_arg) {
            out += "static constexpr op::rt::kernel::RuntimeParam " + param.name() + "{};\n";
            continue;
        }
        out += "__constant__ ";
        out += op::rt::jit::cxx_type(param.type());
        out += " " + param.name();
        if (param.array())
            out += "[" + std::to_string(param.elems()) + "]";
        out += ";\n";
        symbols.push_back({i, param.name()});
    }
    return out;
}

} // namespace

Compiler::Job Compiler::job(const op::rt::jit::ParamSet& params,
                            const KernelSource& source) const {
    std::vector<RuntimeSymbol> symbols;
    std::vector<int> baked_globals;
    for (std::size_t i = 0; i < params.size(); ++i)
        if (params[i].source() == op::rt::abi::ParamSource::array_arg &&
            params[i].mode() == op::rt::jit::ParamMode::baked)
            baked_globals.push_back(params[i].index());
    std::string text;
    for (const char *name : m_headers.names)
        text += std::string{"#include <"} + name + ">\n";
    text += "\n" + op::rt::jit::format_cxx(params) + runtime_declarations(params, symbols) +
            std::string{source.source};
    return {this, std::move(text), std::string{source.name}, std::move(symbols),
            std::move(baked_globals)};
}

std::unique_ptr<JitKernel> Compiler::Job::operator()() {
    const auto& headers = compiler->m_headers;
    gpuRtcProgram_t prog;
    GPU_RTC_CHECK(gpuRtcCreateProgram(&prog, text.c_str(), name.c_str(),
                                      static_cast<int>(headers.data.size()),
                                      headers.data.data(), headers.names.data()));

#ifdef OP2_CUDA
    const char *opts[] = {
        device().arch.c_str(),
        "--std=c++20",
#if CUDA_VERSION >= 12040
        "--minimal",
#endif
        "--device-as-default-execution-space"
    };
#else // OP2_HIP
    const char *opts[] = {
        "--std=c++20",
        "-O3",
        "-munsafe-fp-atomics"
    };
#endif
    auto success =
        gpuRtcCompileProgram(prog, static_cast<int>(sizeof(opts) / sizeof(opts[0])), opts);
    if (success != GPURTC_SUCCESS) {
        size_t log_size;
        GPU_RTC_CHECK(gpuRtcGetProgramLogSize(prog, &log_size));
        if (log_size > 1) {
            std::string log(log_size, '\0');
            GPU_RTC_CHECK(gpuRtcGetProgramLog(prog, log.data()));
            std::printf("%s\n", log.c_str());
        }
        std::fprintf(stderr, "error: JIT compile of %s failed (in %s)\n", name.c_str(),
                     compiler->m_loop.c_str());
        GPU_RTC_CHECK(gpuRtcDestroyProgram(&prog));
        return nullptr;
    }

    size_t cubin_size;
    GPU_RTC_CHECK(gpuRtcGetCodeSize(prog, &cubin_size));
    char *cubin = new char[cubin_size];
    GPU_RTC_CHECK(gpuRtcGetCode(prog, cubin));
    GPU_RTC_CHECK(gpuRtcDestroyProgram(&prog));
    return std::make_unique<JitKernel>(cubin, name, std::move(symbols),
                                       std::move(baked_globals));
}

} // namespace op::rt::gpu
