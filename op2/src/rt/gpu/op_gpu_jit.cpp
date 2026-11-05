#include <op2/rt/gpu/jit.h>

#include <op2/gpu/check.h>

#include <cstddef>
#include <cstdio>
#include <iterator>
#include <utility>

namespace op::rt::gpu {

namespace {

// The run-time parameters as a built kernel has them: a global as its
// wrapper's argument, which the source reads through
// op::rt::kernel::gpu::param_value; any other as a symbol of its module, added to
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
            out += "static constexpr op::rt::kernel::gpu::RuntimeParam ";
            out += param.name();
            out += "{};\n";
            continue;
        }
        out += "__constant__ ";
        out += op::rt::jit::cxx_type(param.type());
        out += " ";
        out += param.name();
        if (param.array())
            out += "[" + std::to_string(param.elems()) + "]";
        out += ";\n";
        symbols.push_back({i, param.name()});
    }
    return out;
}

} // namespace

Compiler::Job Compiler::job(const op::rt::jit::ParamSet& params, std::span<const int> values,
                            const abi::gpu::Family& family) const {
    std::vector<RuntimeSymbol> symbols;
    std::vector<int> baked_globals;
    for (std::size_t i = 0; i < params.size(); ++i)
        if (params[i].source() == op::rt::abi::ParamSource::array_arg &&
            params[i].mode() == op::rt::jit::ParamMode::baked)
            baked_globals.push_back(params[i].index());
    std::string text;
    for (const char *name : m_headers.names)
        text += std::string{"#include <"} + name + ">\n";
    text += "\n" + op::rt::jit::format_cxx(params, values) +
            runtime_declarations(params, symbols);
    if (m_translated != nullptr)
        text += m_translated;
    text += family.source;
    return {m_loop, m_headers, m_arch, std::move(text), std::string{family.kernel},
            std::move(symbols), std::move(baked_globals)};
}

std::unique_ptr<JitKernel> Compiler::Job::operator()() {
    gpuRtcProgram_t prog;
    GPU_RTC_CHECK(gpuRtcCreateProgram(&prog, text.c_str(), name.c_str(),
                                      static_cast<int>(headers.data.size()),
                                      headers.data.data(), headers.names.data()));

#ifdef OP2_CUDA
    const char *opts[] = {
        arch.c_str(),
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
    auto success = gpuRtcCompileProgram(prog, static_cast<int>(std::size(opts)), opts);
    if (success != GPURTC_SUCCESS) {
        std::size_t log_size;
        GPU_RTC_CHECK(gpuRtcGetProgramLogSize(prog, &log_size));
        if (log_size > 1) {
            std::string log(log_size, '\0');
            GPU_RTC_CHECK(gpuRtcGetProgramLog(prog, log.data()));
            std::fprintf(stderr, "%s\n", log.c_str());
        }
        std::fprintf(stderr, "error: JIT compile of %s failed (in %s)\n", name.c_str(),
                     loop.c_str());
        GPU_RTC_CHECK(gpuRtcDestroyProgram(&prog));
        return nullptr;
    }

    std::size_t code_size;
    GPU_RTC_CHECK(gpuRtcGetCodeSize(prog, &code_size));
    std::vector<char> code(code_size);
    GPU_RTC_CHECK(gpuRtcGetCode(prog, code.data()));
    GPU_RTC_CHECK(gpuRtcDestroyProgram(&prog));
    return std::make_unique<JitKernel>(std::move(code), name, std::move(symbols),
                                       std::move(baked_globals));
}

} // namespace op::rt::gpu
