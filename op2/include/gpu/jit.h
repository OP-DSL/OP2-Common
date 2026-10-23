#pragma once

// The CUDA and HIP JIT mechanism: the offline kernels and the device symbols
// they read their parameters from, the kernels built through NVRTC or
// hipRTC, how those have their run-time parameters, and their launches.  The
// policy, the parameters and the build cache are loop/jit.h's.

#include <gpu/check.h>
#include <gpu/config.h>
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
    std::uint64_t uploaded = 0;
};

// A kernel built at run time, loaded on its first launch.
class JitKernel {
private:
    std::string m_name;
    std::vector<RuntimeSymbol> m_symbols;
    bool m_loaded = false;
    int m_max_dynamic_shared_bytes = 0;
    char *m_cubin;
    gpuDrvModule_t m_module;
    gpuDrvFunction_t m_kernel;

    void ensure_loaded() {
        if (m_loaded)
            return;
        GPU_DRV_CHECK(gpuDrvModuleLoadData(&m_module, m_cubin));
        GPU_DRV_CHECK(gpuDrvModuleGetFunction(&m_kernel, m_module, m_name.c_str()));
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
    JitKernel(char *cubin, std::string_view name, std::vector<RuntimeSymbol> symbols)
        : m_name{name}, m_symbols{std::move(symbols)}, m_cubin{cubin} {}
    ~JitKernel() { delete[] m_cubin; }

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
        if (op::loop::jit_policy().debug)
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

    // The prior for when a specialised kernel pays: more registers than
    // OP_JIT_MIN_REGISTERS.
    bool worth_specialising() const {
        return m_attributes.numRegs > config().min_registers;
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
        if (op::loop::jit_policy().debug)
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

// The slot of each value a strategy fixes for its kernel, as the generated
// code names it.
enum class ParamSource {
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
        // when given, else hash.
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

// Builds a loop's specialised kernels through NVRTC or hipRTC.
class Compiler {
private:
    std::string m_loop;
    JitHeaders m_headers;

    // The run-time parameters as a built kernel has them: a scalar global as
    // its wrapper's argument, which the source reads through
    // f2c::param_value; any other as a symbol of its module, added to
    // symbols.
    static std::string runtime_declarations(const op::loop::ParamSet& params,
                                            std::vector<RuntimeSymbol>& symbols) {
        std::string out;
        for (std::size_t i = 0; i < params.size(); ++i) {
            const auto& param = params[i];
            if (param.mode() != op::loop::ParamMode::runtime)
                continue;
            if (param.source() == op::loop::ParamSource::scalar_arg) {
                out += "static constexpr f2c::RuntimeParam " + param.name() + "{};\n";
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
                                       std::vector<RuntimeSymbol> symbols) const {
        gpuRtcProgram_t prog;
        GPU_RTC_CHECK(gpuRtcCreateProgram(
            &prog, text.c_str(), name.c_str(),
            static_cast<int>(m_headers.data.size()), m_headers.data.data(),
            m_headers.names.data()));

#ifdef OP2_CUDA
        const char *opts[] = {
            config().arch.c_str(),
            "--std=c++20",
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
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
        auto success = gpuRtcCompileProgram(prog, sizeof(opts) / sizeof(char *), opts);
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
        return std::make_unique<JitKernel>(cubin, name, std::move(symbols));
    }

public:
    Compiler(std::string_view loop, JitHeaders headers)
        : m_loop{loop}, m_headers{std::move(headers)} {}

    // The job that builds source's kernel with the parameters' current
    // values baked in, or at run time as their modes say; it prints the
    // compiler's log and returns none when the compile fails.
    auto job(const op::loop::ParamSet& params, const KernelSource& source) const {
        std::vector<RuntimeSymbol> symbols;
        std::string text = std::string("#include <kernel/prelude.h>\n"
                                       "#include <kernel/strategy.h>\n") +
                           (m_headers.params ? "#include <op_f2c_params.h>\n" : "") +
                           "\nnamespace f2c = op::f2c;\n" +
                           op::loop::format_cxx(params) +
                           runtime_declarations(params, symbols) +
                           std::string{source.source};
        return [this, text = std::move(text), name = std::string{source.name},
                symbols = std::move(symbols)]() mutable {
            return compile(text, name, std::move(symbols));
        };
    }
};

// --- The loop's JIT -----------------------------------------------------------------

// A loop's parameters, the symbols its offline kernels read them from, and
// the compiler of its specialised kernels.
class Jit {
private:
    op::loop::ParamSet m_params;
    SymbolBindings m_symbols;
    Compiler m_compiler;

    template<typename T>
    void bind(std::size_t param, const T *symbol, std::uint64_t *shared_hash) {
        if (symbol == nullptr)
            return;
        void *data_d = nullptr;
        GPU_CHECK(gpuGetSymbolAddress(&data_d, (const void *)symbol));
        m_symbols.bind(param, data_d, shared_hash);
    }

public:
    Jit(std::string_view loop, JitHeaders headers)
        : m_params{loop}, m_compiler{loop, std::move(headers)} {}
    Jit(const Jit&) = delete;

    // --- Parameters, registered by the generated code ---

    // Each kind a built kernel can have at run time but a dat's dimension
    // and a strategy's value, which its source needs as constants.

    template<typename T>
    void add_param(std::string_view name, T *data, const T *symbol = nullptr,
                   uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add(name, data, 1, op::loop::ParamSource::external, -1, true),
             symbol, hash_device_ptr);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, std::size_t len,
                   const T *symbol = nullptr, uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add(name, data, len, op::loop::ParamSource::external, -1, true),
             symbol, hash_device_ptr);
    }

    template<typename T>
    void add_scalar_arg_param(std::string_view name, T *data, int arg_index,
                              const T *symbol = nullptr,
                              uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add(name, data, 1, op::loop::ParamSource::scalar_arg, arg_index,
                          true),
             symbol, hash_device_ptr);
    }

    void add_dat_stride_param(std::string_view name, int *data, int arg_index,
                              const int *symbol = nullptr,
                              uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add(name, data, 1, op::loop::ParamSource::dat_stride, arg_index,
                          true),
             symbol, hash_device_ptr);
    }

    void add_dat_dim_param(std::string_view name, int *data, int arg_index) {
        m_params.add(name, data, 1, op::loop::ParamSource::dat_dim, arg_index);
    }

    // A value each strategy decides (global stride, block size, singles,
    // wavefront), given to every acquire and upload; symbol is the copy an
    // offline kernel reads, if any.
    void add_launch_param(std::string_view name, ParamSource source,
                          const int *symbol = nullptr) {
        bind(m_params.add(name, static_cast<int *>(nullptr), 1,
                          op::loop::ParamSource::strategy, static_cast<int>(source)),
             symbol, nullptr);
    }

    // --- Per call ---

    // The call's values, at invoke.
    void update(op_arg *args, int nargs) { m_params.update(args, nargs); }

    // Upload the changed values an offline kernel reads, under a strategy's
    // launch values.
    void upload(const LaunchValues& launch) {
        m_params.apply(launch.slots());
        m_symbols.upload(m_params);
    }

    op::loop::ParamSet& params() { return m_params; }
    const Compiler& compiler() const { return m_compiler; }
};

} // namespace op::gpu
