#pragma once

// The CUDA and HIP JIT mechanism: the offline kernels and the device symbols
// they read their parameters from, the kernels built through NVRTC or
// hipRTC, how those have their run-time parameters, and their launches.  The
// policy, the parameters and the build cache are loop/jit.h's.

#include <gpu/check.h>
#include <gpu/options.h>
#include <gpu/shims.h>
#include <loop/jit.h>
#include <op_lib_core.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace op::gpu {

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

// A kernel's resources as its compiler left them: registers and local
// memory per thread, static shared memory per block, and the most threads a
// block can have with them.
struct KernelFigures {
    int registers = 0;
    int local_bytes = 0;
    int shared_bytes = 0;
    int max_threads = 0;

    // "64 regs", with the local and shared memory when there is any.
    std::string text() const {
        std::string out = std::to_string(registers) + " regs";
        if (local_bytes > 0)
            out += ", " + std::to_string(local_bytes) + " B local";
        if (shared_bytes > 0)
            out += ", " + std::to_string(shared_bytes) + " B shared";
        return out;
    }
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

    void ensure_loaded() {
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
            GPU_DRV_CHECK(gpuDrvModuleGetGlobal(&symbol.address, &bytes, m_module,
                                                symbol.name.c_str()));
        }
        m_loaded = true;
        delete[] m_cubin;
        m_cubin = nullptr;
    }

public:
    JitKernel(const JitKernel&) = delete;
    JitKernel(char *cubin, std::string_view name, std::vector<RuntimeSymbol> symbols,
              std::vector<int> baked_globals)
        : m_name{name}, m_symbols{std::move(symbols)},
          m_baked_globals{std::move(baked_globals)}, m_cubin{cubin} {}
    ~JitKernel() { delete[] m_cubin; }

    const std::vector<int>& baked_globals() const { return m_baked_globals; }

    const KernelFigures& figures() {
        ensure_loaded();
        return m_figures;
    }

    // Launch, its module's symbols first given the call's values of params.
    void launch(int num_blocks, int block_size, void **args, int shared_bytes,
                const op::loop::ParamSet& params) {
        ensure_loaded();
        for (auto& symbol : m_symbols) {
            const auto& param = params[symbol.param];
            if (param.last_hash() == symbol.uploaded)
                continue;
            GPU_DRV_CHECK(gpuDrvMemcpyHtoDAsync(symbol.address, param.data(),
                                                param.bytes(), 0));
            symbol.uploaded = param.last_hash();
        }
        if (shared_bytes > m_max_dynamic_shared_bytes) {
            GPU_DRV_CHECK(gpuDrvFuncSetAttribute(
                m_kernel, gpuDrvFuncAttributeMaxDynamicSharedMemorySize,
                shared_bytes));
            m_max_dynamic_shared_bytes = shared_bytes;
        }
        GPU_DRV_CHECK(gpuDrvLaunchKernel(m_kernel, num_blocks, 1, 1, block_size, 1,
                                         1, shared_bytes, NULL, args, 0));
        GPU_CHECK(gpuPeekAtLastError());
        if (op::loop::jit_options().debug)
            GPU_CHECK(gpuStreamSynchronize(0));
    }
};

// A strategy's kernel compiled with the application, if any.
class OfflineKernel {
private:
    const void *m_function;
    gpuFuncAttributes_t m_attributes{};
    int m_max_dynamic_shared_bytes = 0;

public:
    explicit OfflineKernel(const void *function) : m_function{function} {
        if (function != nullptr)
            GPU_CHECK(gpuFuncGetAttributes(&m_attributes, function));
    }

    // Whether it exists and can launch block_size threads.
    bool usable(int block_size) const {
        return m_function != nullptr && block_size <= m_attributes.maxThreadsPerBlock;
    }

    int registers() const { return m_attributes.numRegs; }

    KernelFigures figures() const {
        return {m_attributes.numRegs, static_cast<int>(m_attributes.localSizeBytes),
                static_cast<int>(m_attributes.sharedSizeBytes),
                m_attributes.maxThreadsPerBlock};
    }

    void launch(int num_blocks, int block_size, void **args, int shared_bytes) {
        assert(m_function != nullptr);
        if (shared_bytes > m_max_dynamic_shared_bytes) {
            GPU_CHECK(gpuFuncSetAttribute(
                m_function, gpuFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            m_max_dynamic_shared_bytes = shared_bytes;
        }
        GPU_CHECK(gpuLaunchKernel(m_function, num_blocks, block_size, args,
                                  shared_bytes, 0));
        GPU_CHECK(gpuPeekAtLastError());
        if (op::loop::jit_options().debug)
            GPU_CHECK(gpuStreamSynchronize(0));
    }
};

// What a strategy's kernel is built from: the offline function, if any, and
// the source a specialised one is compiled from.
struct KernelSource {
    std::string_view name;
    const void *offline = nullptr;
    std::string_view source;
};

// --- Parameters ---------------------------------------------------------------------

// Each value a strategy fixes for its kernel, by its slot.
enum class LaunchValue {
    global_stride,
    block_size,
    unstaged_singles,
    wavefront,
};

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

    void upload(op::loop::ParamSet& params) {
        for (auto& binding : m_bindings) {
            auto& param = params[binding.param];
            auto& uploaded =
                binding.shared_hash != nullptr ? *binding.shared_hash : binding.hash;
            auto hash = param.hash();
            if (hash == uploaded)
                continue;
            GPU_CHECK(gpuMemcpyAsync(binding.data_d, param.data(), param.bytes(),
                                     gpuMemcpyHostToDevice));
            uploaded = hash;
        }
    }
};

// --- Builds -------------------------------------------------------------------------

// The headers a compile includes, embedded in the translation unit.
struct JitHeaders {
    std::vector<const char *> data;
    std::vector<const char *> names;
    // Whether op_f2c_params.h, the application's constants, is among them.
    bool params = false;
};

// How a strategy compiles its specialised kernels beyond the parameters'
// values: with its block size as launch bounds, and spilling registers to
// shared memory (kernel/strategy.h's OP2_JIT_* macros).
struct CompileSwitches {
    bool launch_bounds = false;
    bool smem_spilling = false;
};

// Builds a loop's specialised kernels through NVRTC or hipRTC.
class Compiler {
private:
    std::string m_loop;
    JitHeaders m_headers;

    // The run-time parameters as a built kernel has them: a global as its
    // wrapper's argument, which the source reads through op::kernel::param_value;
    // any other as a symbol of its module, added to symbols.
    static std::string runtime_declarations(const op::loop::ParamSet& params,
                                            std::vector<RuntimeSymbol>& symbols) {
        std::string out;
        for (std::size_t i = 0; i < params.size(); ++i) {
            const auto& param = params[i];
            if (param.mode() != op::loop::ParamMode::runtime)
                continue;
            if (param.source() == op::loop::ParamSource::scalar_arg ||
                param.source() == op::loop::ParamSource::array_arg) {
                out += "static constexpr op::kernel::RuntimeParam " + param.name() + "{};\n";
                continue;
            }
            out += "__constant__ ";
            out += op::loop::cxx_type(param.type());
            out += " " + param.name();
            if (param.array())
                out += "[" + std::to_string(param.elems()) + "]";
            out += ";\n";
            symbols.push_back({i, param.name()});
        }
        return out;
    }

    std::unique_ptr<JitKernel> compile(const std::string& text, const std::string& name,
                                       CompileSwitches switches,
                                       std::vector<RuntimeSymbol> symbols,
                                       std::vector<int> baked_globals) const {
        gpuRtcProgram_t prog;
        GPU_RTC_CHECK(gpuRtcCreateProgram(
            &prog, text.c_str(), name.c_str(),
            static_cast<int>(m_headers.data.size()), m_headers.data.data(),
            m_headers.names.data()));

#ifdef OP2_CUDA
        std::vector<const char *> opts = {
            device().arch.c_str(),
            "--std=c++20",
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
            "--minimal",
#endif
            "--device-as-default-execution-space"
        };
#else // OP2_HIP
        std::vector<const char *> opts = {
            "--std=c++20",
            "-O3",
            "-munsafe-fp-atomics"
        };
#endif
        if (switches.launch_bounds)
            opts.push_back("-DOP2_LAUNCH_BOUNDS");
        if (switches.smem_spilling)
            opts.push_back("-DOP2_SMEM_SPILLING");
        auto success =
            gpuRtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
        if (success != GPURTC_SUCCESS) {
            size_t log_size;
            GPU_RTC_CHECK(gpuRtcGetProgramLogSize(prog, &log_size));
            if (log_size > 1) {
                std::string log(log_size, '\0');
                GPU_RTC_CHECK(gpuRtcGetProgramLog(prog, log.data()));
                std::printf("%s\n", log.c_str());
            }
            std::fprintf(stderr, "error: JIT compile of %s failed (in %s)\n",
                         name.c_str(), m_loop.c_str());
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

public:
    Compiler(std::string_view loop, JitHeaders headers)
        : m_loop{loop}, m_headers{std::move(headers)} {}

    // The job that builds source's kernel with the parameters' current
    // values baked in, or at run time as their modes say, under switches; it
    // prints the compiler's log and returns none when the compile fails.
    auto job(const op::loop::ParamSet& params, const KernelSource& source,
             CompileSwitches switches) const {
        std::vector<RuntimeSymbol> symbols;
        std::vector<int> baked_globals;
        for (std::size_t i = 0; i < params.size(); ++i)
            if (params[i].source() == op::loop::ParamSource::array_arg &&
                params[i].mode() == op::loop::ParamMode::baked)
                baked_globals.push_back(params[i].index());
        std::string text = std::string("#include <kernel/prelude.h>\n"
                                       "#include <kernel/strategy.h>\n") +
                           (m_headers.params ? "#include <op_f2c_params.h>\n" : "") +
                           "\n" +
                           op::loop::format_cxx(params) +
                           runtime_declarations(params, symbols) +
                           std::string{source.source};
        return [this, text = std::move(text), name = std::string{source.name}, switches,
                symbols = std::move(symbols),
                baked_globals = std::move(baked_globals)]() mutable {
            return compile(text, name, switches, std::move(symbols),
                           std::move(baked_globals));
        };
    }
};

} // namespace op::gpu
