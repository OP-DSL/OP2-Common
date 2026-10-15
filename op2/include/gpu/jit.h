#pragma once

// The JIT service of the CUDA and HIP backend: the parameters a loop's
// kernels are specialised on, the compiles through NVRTC or hipRTC, and
// every JIT policy, from the enable and force variables to the minimum-calls
// delay and the register prior.  One per loop.

#include <extern/rapidhash.h>
#include <gpu/config.h>
#include <gpu/shims.h>
#include <loop/strategy.h>
#include <op_rt_support.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#define NVRTC_SAFE_CALL(x)                                                      \
    do {                                                                        \
        gpuRtcResult_t result = x;                                              \
        if (result != GPURTC_SUCCESS) {                                         \
            const char *msg = gpuRtcGetErrorString(result);                     \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,     \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUDA_SAFE_CALL(x)                                                       \
    do {                                                                        \
        gpuError_t result = x;                                                  \
        if (result != gpuSuccess) {                                             \
            const char *msg = gpuGetErrorString(result);                        \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,     \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#ifdef OP2_CUDA
#define CU_SAFE_CALL(x)                                                         \
    do {                                                                        \
        gpuDrvResult_t result = x;                                              \
        if (result != GPU_SUCCESS) {                                            \
            const char *msg;                                                    \
            gpuDrvGetErrorName(result, &msg);                                   \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,     \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)
#else
#define CU_SAFE_CALL(x) CUDA_SAFE_CALL(x)
#endif

namespace op::gpu {

// --- Hashing ------------------------------------------------------------------------

constexpr uint64_t hash_seed_default = RAPID_SEED;

template<typename T>
inline uint64_t hash(const T key, uint64_t seed = hash_seed_default) {
    return rapidhash_withSeed((void *)&key, sizeof(T), seed);
}

inline uint64_t hash_bytes(const void *key, size_t len,
                           uint64_t seed = hash_seed_default) {
    return rapidhash_withSeed(key, len, seed);
}

// --- Kernels --------------------------------------------------------------------------

class JitKernel {
private:
    std::string m_name;
    bool m_loaded = false;
    int m_max_dynamic_shared_bytes = 0;
    char *m_cubin;
    gpuDrvModule_t m_module;
    gpuDrvFunction_t m_kernel;

    void ensure_loaded() {
        if (m_loaded)
            return;
        CU_SAFE_CALL(gpuDrvModuleLoadData(&m_module, m_cubin));
        CU_SAFE_CALL(gpuDrvModuleGetFunction(&m_kernel, m_module, m_name.c_str()));
        m_loaded = true;
        delete[] m_cubin;
        m_cubin = nullptr;
    }

public:
    JitKernel(const JitKernel&) = delete;
    JitKernel(char *cubin, std::string_view name) : m_name{name}, m_cubin{cubin} {}

    void launch(int num_blocks, int block_size, void **args, int shared_bytes) {
        ensure_loaded();
        if (shared_bytes > m_max_dynamic_shared_bytes) {
            CU_SAFE_CALL(gpuDrvFuncSetAttribute(
                m_kernel, gpuDrvFuncAttributeMaxDynamicSharedMemorySize,
                shared_bytes));
            m_max_dynamic_shared_bytes = shared_bytes;
        }
        CU_SAFE_CALL(gpuDrvLaunchKernel(m_kernel, num_blocks, 1, 1, block_size, 1,
                                        1, shared_bytes, NULL, args, 0));
        CUDA_SAFE_CALL(gpuPeekAtLastError());
        if (config().debug)
            CUDA_SAFE_CALL(gpuStreamSynchronize(0));
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

enum class ParamType {
    i32,
    i64,
    f32,
    f64,
    logical,
};

// Where a parameter's value comes from: the call, the strategy's launch, or
// nothing after registration.
enum class ParamSource {
    external,       // a constant the application sets
    scalar_arg,     // a scalar global argument
    dat_stride,     // a dat's SoA stride
    dat_dim,        // a dat's run-time dimension
    global_stride,  // the launch's per-thread global stride
    block_size,     // the launch's block size
    setting,        // a run-wide setting, fixed at registration
};

template<typename T> struct JitTypes {};
template<> struct JitTypes<int>      { static const ParamType value = ParamType::i32; };
template<> struct JitTypes<int64_t>  { static const ParamType value = ParamType::i64; };
template<> struct JitTypes<float>    { static const ParamType value = ParamType::f32; };
template<> struct JitTypes<double>   { static const ParamType value = ParamType::f64; };
template<> struct JitTypes<bool>     { static const ParamType value = ParamType::logical; };

class JitParam {
private:
    std::string m_name;
    void *m_data;
    void *m_data_d;
    std::size_t m_n_elems;
    std::size_t m_elem_size;
    ParamType m_type;
    bool m_array;
    ParamSource m_source;
    int m_arg_index;
    uint64_t m_hash_last = 0;
    uint64_t m_hash_device = 0;
    uint64_t *m_hash_device_ptr = nullptr;

public:
    // A parameter of len elements, an array when len is more than one, read
    // from data and uploaded to data_d when that is given.
    template<typename T>
    JitParam(std::string_view name, T *data, std::size_t len, T *data_d,
             uint64_t *hash_device_ptr, ParamSource source, int arg_index)
        : m_name{name}, m_data{data}, m_data_d{data_d}, m_n_elems{len},
          m_elem_size{sizeof(T)}, m_type{JitTypes<T>::value}, m_array{len > 1},
          m_source{source}, m_arg_index{arg_index},
          m_hash_device_ptr{hash_device_ptr} {}

    ParamSource source() const { return m_source; }

    // The call's values.
    void update(op_arg *args, int nargs) {
        switch (m_source) {
        case ParamSource::scalar_arg:
            assert(m_arg_index >= 0 && m_arg_index < nargs);
            assert(args[m_arg_index].data != nullptr);
            std::memcpy(m_data, args[m_arg_index].data, m_elem_size);
            return;
        case ParamSource::dat_stride: {
            assert(m_arg_index >= 0 && m_arg_index < nargs);
            assert(m_elem_size == sizeof(int) && m_n_elems == 1);
            int size = getSetSizeFromOpArg(&args[m_arg_index]);
            *static_cast<int *>(m_data) = (size + 31) & ~31;
            return;
        }
        case ParamSource::dat_dim:
            assert(m_arg_index >= 0 && m_arg_index < nargs);
            assert(m_elem_size == sizeof(int) && m_n_elems == 1);
            *static_cast<int *>(m_data) = args[m_arg_index].dim;
            return;
        default:
            (void)nargs;
            return;
        }
    }

    // The launch's values, which the strategy decides.
    void set_launch(int global_stride, int block_size) {
        if (m_source == ParamSource::global_stride)
            *static_cast<int *>(m_data) = global_stride;
        else if (m_source == ParamSource::block_size)
            *static_cast<int *>(m_data) = block_size;
    }

    uint64_t hash() {
        m_hash_last = hash_bytes(m_data, m_n_elems * m_elem_size);
        return m_hash_last;
    }

    // Upload to the device symbol an offline kernel reads, when changed.
    void upload() {
        if (m_data_d == nullptr)
            return;
        auto hash_device = m_hash_device_ptr != nullptr ? *m_hash_device_ptr
                                                        : m_hash_device;
        if (hash() == hash_device)
            return;
        CUDA_SAFE_CALL(gpuMemcpyAsync(m_data_d, m_data, m_elem_size * m_n_elems,
                                      gpuMemcpyHostToDevice));
        if (m_hash_device_ptr != nullptr)
            *m_hash_device_ptr = m_hash_last;
        else
            m_hash_device = m_hash_last;
    }

    std::string format_type() const {
        switch (m_type) {
        case ParamType::i32:     return "int";
        case ParamType::i64:     return "int64_t";
        case ParamType::f32:     return "float";
        case ParamType::f64:     return "double";
        case ParamType::logical: return "bool";
        }
        __builtin_unreachable();
    }

    std::string format_value() const {
        std::ostringstream os;
        if (m_array)
            os << "{ ";
        for (std::size_t i = 0; i < m_n_elems; ++i) {
            char *elem = (char *)m_data + m_elem_size * i;
            switch (m_type) {
            case ParamType::i32:     os << *((int *)elem); break;
            case ParamType::i64:     os << *((int64_t *)elem); break;
            case ParamType::f32:     os << std::hexfloat << *((float *)elem); break;
            case ParamType::f64:     os << std::hexfloat << *((double *)elem); break;
            case ParamType::logical: os << std::boolalpha << *((bool *)elem); break;
            }
            if (m_array && i < m_n_elems - 1)
                os << ", ";
        }
        if (m_array)
            os << " }";
        return os.str();
    }

    // The constant a specialised kernel is compiled with.
    std::string format() const {
        std::ostringstream os;
        os << "static constexpr " << format_type() << " " << m_name;
        if (m_array)
            os << "[" << m_n_elems << "]";
        os << " = " << format_value() << ";" << std::endl;
        return os.str();
    }
};

// --- The service --------------------------------------------------------------------

// The headers a compile includes, embedded in the translation unit.
struct JitHeaders {
    std::vector<const char *> data;
    std::vector<const char *> names;
    // Whether op_f2c_params.h, the application's constants, is among them.
    bool params = false;
};

// A strategy's compiled kernels, one per parameter hash, and its offline
// kernel's attributes.
class JitKernels {
    friend class JitService;

private:
    struct Compile {
        std::size_t count = 0;
        bool started = false;
        std::thread thread;
    };

    gpuFuncAttributes_t m_offline_attrs{};
    int m_offline_max_dynamic_shared_bytes = 0;
    std::unordered_map<uint64_t, JitKernel> m_kernels;
    std::unordered_map<uint64_t, Compile> m_compiles;
    std::mutex m_mutex;

public:
    JitKernels() = default;
    JitKernels(const JitKernels&) = delete;
    ~JitKernels() { join(); }

    void register_offline(const void *kernel) {
        if (kernel != nullptr)
            CUDA_SAFE_CALL(gpuFuncGetAttributes(&m_offline_attrs, kernel));
    }

    // Launch the offline kernel.
    void launch_offline(const void *kernel, int num_blocks, int block_size,
                        void **args, int shared_bytes) {
        assert(kernel != nullptr);
        if (shared_bytes > m_offline_max_dynamic_shared_bytes) {
            CUDA_SAFE_CALL(gpuFuncSetAttribute(
                kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            m_offline_max_dynamic_shared_bytes = shared_bytes;
        }
        CUDA_SAFE_CALL(gpuLaunchKernel(kernel, num_blocks, block_size, args,
                                       shared_bytes, 0));
        CUDA_SAFE_CALL(gpuPeekAtLastError());
        if (config().debug)
            CUDA_SAFE_CALL(gpuStreamSynchronize(0));
    }

    // Wait for every compile still running.
    void join() {
        for (auto& [hash, compile] : m_compiles)
            if (compile.thread.joinable())
                compile.thread.join();
    }
};

struct Acquired {
    op::loop::Status status = op::loop::Status::unavailable;
    // The specialised kernel; null to run the offline one.
    JitKernel *kernel = nullptr;
};

class JitService {
private:
    std::string m_loop;
    JitHeaders m_headers;
    std::vector<JitParam> m_params;
    static inline std::atomic_int active_threads = 0;

    uint64_t hash_params() {
        uint64_t hash = hash_seed_default;
        for (auto& param : m_params)
            hash = op::gpu::hash(param.hash(), hash);
        return hash;
    }

    std::string format_params() const {
        std::string src;
        for (const auto& param : m_params)
            src += param.format();
        return src;
    }

    std::thread compile(JitKernels& kernels, const KernelSource& source,
                        uint64_t hash) {
        ++active_threads;
        std::string jit_src = std::string("#include <kernel/prelude.h>\n"
                                          "#include <kernel/strategy.h>\n") +
                              (m_headers.params ? "#include <op_f2c_params.h>\n" : "") +
                              "\nnamespace f2c = op::f2c;\n" + format_params() +
                              std::string{source.source};

        auto do_compile = [this, &kernels, name = std::string{source.name}](
                              std::string jit_src, uint64_t hash) {
            gpuRtcProgram_t prog;
            NVRTC_SAFE_CALL(gpuRtcCreateProgram(
                &prog, jit_src.c_str(), name.c_str(),
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
                NVRTC_SAFE_CALL(gpuRtcGetProgramLogSize(prog, &log_size));
                if (log_size > 1) {
                    std::string log(log_size, '\0');
                    NVRTC_SAFE_CALL(gpuRtcGetProgramLog(prog, log.data()));
                    std::printf("%s\n", log.c_str());
                }
                std::fprintf(stderr, "error: JIT compile of %s failed (in %s)\n",
                             name.c_str(), m_loop.c_str());
                std::exit(1);
            }

            size_t cubin_size;
            NVRTC_SAFE_CALL(gpuRtcGetCodeSize(prog, &cubin_size));
            char *cubin = new char[cubin_size];
            NVRTC_SAFE_CALL(gpuRtcGetCode(prog, cubin));
            NVRTC_SAFE_CALL(gpuRtcDestroyProgram(&prog));

            std::scoped_lock lock(kernels.m_mutex);
            auto [it, inserted] = kernels.m_kernels.emplace(
                std::piecewise_construct, std::forward_as_tuple(hash),
                std::forward_as_tuple(cubin, name));
            assert(inserted);
            --active_threads;
        };

        return std::thread(do_compile, std::move(jit_src), hash);
    }

    template<typename T>
    static T *lookup_symbol(const T *symbol) {
        if (symbol == nullptr)
            return nullptr;
        T *data_d = nullptr;
        CUDA_SAFE_CALL(gpuGetSymbolAddress((void **)&data_d, (const void *)symbol));
        return data_d;
    }

public:
    JitService(std::string_view loop, JitHeaders headers)
        : m_loop{loop}, m_headers{std::move(headers)} {}
    JitService(const JitService&) = delete;

    // --- Parameters, registered by the generated code ---

    template<typename T>
    void add_param(std::string_view name, T *data, const T *symbol = nullptr,
                   uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, 1, lookup_symbol(symbol),
                              hash_device_ptr, ParamSource::external, -1);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, std::size_t len,
                   const T *symbol = nullptr, uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, len, lookup_symbol(symbol),
                              hash_device_ptr, ParamSource::external, -1);
    }

    template<typename T>
    void add_scalar_arg_param(std::string_view name, T *data, int arg_index,
                              const T *symbol = nullptr,
                              uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, 1, lookup_symbol(symbol),
                              hash_device_ptr, ParamSource::scalar_arg, arg_index);
    }

    void add_dat_stride_param(std::string_view name, int *data, int arg_index,
                              const int *symbol = nullptr,
                              uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, 1, lookup_symbol(symbol),
                              hash_device_ptr, ParamSource::dat_stride, arg_index);
    }

    void add_dat_dim_param(std::string_view name, int *data, int arg_index) {
        m_params.emplace_back(name, data, 1, static_cast<int *>(nullptr), nullptr,
                              ParamSource::dat_dim, arg_index);
    }

    void add_global_stride_param(std::string_view name, int *data,
                                 const int *symbol = nullptr,
                                 uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, 1, lookup_symbol(symbol),
                              hash_device_ptr, ParamSource::global_stride, -1);
    }

    void add_block_size_param(std::string_view name, int *data) {
        m_params.emplace_back(name, data, 1, static_cast<int *>(nullptr), nullptr,
                              ParamSource::block_size, -1);
    }

    void add_setting_param(std::string_view name, int *data) {
        m_params.emplace_back(name, data, 1, static_cast<int *>(nullptr), nullptr,
                              ParamSource::setting, -1);
    }

    // --- Per call ---

    // The call's values, at invoke.
    void update(op_arg *args, int nargs) {
        for (auto& param : m_params)
            param.update(args, nargs);
    }

    // The launch's values, set by a strategy before it asks for its kernel.
    void set_launch(int global_stride, int block_size) {
        for (auto& param : m_params)
            param.set_launch(global_stride, block_size);
    }

    // Upload the changed values the offline kernels read.
    void upload() {
        for (auto& param : m_params)
            param.upload();
    }

    // Whether the offline kernel can launch at a block size.
    static bool offline_usable(const JitKernels& kernels,
                               const KernelSource& source, int block_size) {
        return source.offline != nullptr &&
               block_size <= kernels.m_offline_attrs.maxThreadsPerBlock;
    }

    // The kernel for the current values: the offline one, unless the policy
    // specialises (no usable offline kernel, OP_JIT_FORCE, or more registers
    // than OP_JIT_MIN_REGISTERS); then the compiled one once it exists, its
    // compile starting once asked OP_JIT_MIN_CALLS times, or at once when
    // now.  pending while nothing can run, unavailable when nothing ever can.
    Acquired acquire(JitKernels& kernels, const KernelSource& source,
                     int block_size, bool now) {
        const auto& run = config();
        bool offline = offline_usable(kernels, source, block_size);
        if (!run.enable)
            return {offline ? op::loop::Status::ready : op::loop::Status::unavailable,
                    nullptr};
        bool specialise = !offline || run.force ||
                          kernels.m_offline_attrs.numRegs > run.min_registers;
        if (!specialise)
            return {op::loop::Status::ready, nullptr};

        auto hash = hash_params();
        {
            std::scoped_lock lock(kernels.m_mutex);
            auto found = kernels.m_kernels.find(hash);
            if (found != kernels.m_kernels.end())
                return {op::loop::Status::ready, &found->second};
        }

        auto& compile = kernels.m_compiles[hash];
        ++compile.count;
        if ((now || compile.count >= static_cast<std::size_t>(run.min_calls)) &&
            !compile.started && active_threads < run.max_threads) {
            if (run.debug)
                std::printf("compiling %s for hash %lx\n",
                            std::string{source.name}.c_str(), hash);
            compile.started = true;
            compile.thread = this->compile(kernels, source, hash);
            if (run.seq_compile) {
                compile.thread.join();
                std::scoped_lock lock(kernels.m_mutex);
                return {op::loop::Status::ready, &kernels.m_kernels.at(hash)};
            }
        }

        return {offline ? op::loop::Status::ready : op::loop::Status::pending,
                nullptr};
    }

    // Whether the current values' kernel is being compiled.
    bool compiling(JitKernels& kernels) {
        auto found = kernels.m_compiles.find(hash_params());
        return found != kernels.m_compiles.end() && found->second.started;
    }
};

} // namespace op::gpu
