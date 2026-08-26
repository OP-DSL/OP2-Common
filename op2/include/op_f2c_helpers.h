#pragma once

#include <extern/rapidhash.h>
#include <op_lib_cpp.h>
#include <op_profile.h>
#include <op_gpu_shims.h>

#include <array>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <string>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <functional>
#include <memory>
// #include <iostream>

extern "C" int getBlockLimit(op_arg *args, int nargs, int block_size,
                              const char *name);

#define NVRTC_SAFE_CALL(x)                                                          \
    do {                                                                            \
        gpuRtcResult_t result = x;                                                  \
        if (result != GPURTC_SUCCESS) {                                             \
            const char *msg = gpuRtcGetErrorString(result);                         \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,         \
                    __FILE__, __LINE__);                                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

#define CUDA_SAFE_CALLN(x)                                                          \
    do {                                                                            \
        gpuError_t result = x;                                                      \
        if (result != gpuSuccess) {                                                 \
            const char *msg = gpuGetErrorString(result);                            \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d (in %s)\n", msg, \
                    __FILE__, __LINE__, m_name.c_str());                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

#define CUDA_SAFE_CALL(x)                                                           \
    do {                                                                            \
        gpuError_t result = x;                                                      \
        if (result != gpuSuccess) {                                                 \
            const char *msg = gpuGetErrorString(result);                            \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,         \
                    __FILE__, __LINE__);                                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

#ifdef OP2_CUDA
#define CU_SAFE_CALL(x)                                                             \
    do {                                                                            \
        gpuDrvResult_t result = x;                                                  \
        if (result != GPU_SUCCESS) {                                                \
            const char *msg;                                                        \
            gpuDrvGetErrorName(result, &msg);                                       \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d (in %s)\n", msg, \
                    __FILE__, __LINE__, m_name.c_str());                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)
#endif

#ifdef OP2_HIP
#define CU_SAFE_CALL(x) CUDA_SAFE_CALL(x)
#endif


namespace op::f2c {

constexpr uint64_t hash_seed_default = RAPID_SEED;

static bool jit_initialized = false;

static bool jit_enable = true;
static bool jit_seq_compile = false;
static bool jit_debug = false;
static bool jit_force = false;

#if defined(OP2_CUDA) && __CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 3
static int jit_max_threads = 16;
#else
// No multi-threaded NVVM/hiprtc but still some gain for multithreading
static int jit_max_threads = 4;
#endif

static std::atomic_int jit_active_threads = 0;

static std::string jit_arch = "";

static void jit_init() {
    if (jit_initialized) return;

    char *enable_str = std::getenv("OP_JIT_ENABLE");
    if (enable_str != nullptr) {
        auto enable = std::string(enable_str);
        std::transform(enable.begin(), enable.end(), enable.begin(),
            [](auto c){ return std::tolower(c); });

        if (enable == "0" || enable == "no" || enable == "false") {
            std::printf("Disabling JIT compilation\n");
            jit_enable = false;
        }
    }

    char *debug_str = std::getenv("OP_JIT_DEBUG");
    if (debug_str != nullptr) {
        auto debug = std::string(debug_str);
        std::transform(debug.begin(), debug.end(), debug.begin(),
            [](auto c){ return std::tolower(c); });

        if (debug == "1" || debug == "yes" || debug == "true") {
            std::printf("Enabling JIT debug\n");
            jit_debug = true;
        }
    }

    char *seq_compile_str = std::getenv("OP_JIT_SEQ_COMPILE");
    if (seq_compile_str != nullptr) {
        auto seq_compile = std::string(seq_compile_str);
        std::transform(seq_compile.begin(), seq_compile.end(), seq_compile.begin(),
            [](auto c){ return std::tolower(c); });

        if (seq_compile == "1" || seq_compile == "yes" || seq_compile == "true")
            jit_seq_compile = true;
    }

    char *force_str = std::getenv("OP_JIT_FORCE");
    if (force_str != nullptr) {
        auto force = std::string(force_str);
        std::transform(force.begin(), force.end(), force.begin(),
            [](auto c){ return std::tolower(c); });

        if (force == "1" || force == "yes" || force == "true") {
            std::printf("Forcing JIT compilation regardless of register count\n");
            jit_force = true;
        }
    }

    char *max_threads_str = getenv("OP_JIT_MAX_THREADS");
    if (max_threads_str != nullptr) {
      int max_threads_int = -1;

      try {
        max_threads_int = std::stoi(max_threads_str);
      } catch (...) {};

      if (max_threads_int < 0)
        std::printf("warning: OP_JIT_MAX_THREADS set to unsupported value: %s\n", max_threads_str);
      else
        jit_max_threads = max_threads_int;
    }

    int device;
    CUDA_SAFE_CALL(gpuGetDevice(&device));

    gpuDeviceProp_t props;
    CUDA_SAFE_CALL(gpuGetDeviceProperties(&props, device));

#ifdef OP2_CUDA
    int cc = props.major * 10 + props.minor;
    jit_arch = "-arch=sm_" + std::to_string(cc);

    if (jit_debug)
        std::printf("JIT arch flag: %s\n", jit_arch.c_str());
#endif

    jit_initialized = true;
}

template<typename T>
static inline uint64_t hash(const T key, uint64_t seed = hash_seed_default) {
    return rapidhash_withSeed((void *)&key, sizeof(T), seed);
}

template<typename T>
static inline uint64_t hash(const T* key, size_t len, uint64_t seed = hash_seed_default) {
    return rapidhash_withSeed((void *)key, sizeof(T) * len, seed);
}

template<>
inline uint64_t hash(const void* key, size_t len, uint64_t seed) {
    return rapidhash_withSeed(key, len, seed);
}

class JitKernel {
private:
    std::string m_name;

    bool m_loaded = false;
    int m_max_dynamic_shared_bytes = 0;
    char *m_cubin;

    gpuDrvModule_t m_module;
    gpuDrvFunction_t m_kernel;

    void ensure_loaded() {
        if (m_loaded) return;

        CU_SAFE_CALL(gpuDrvModuleLoadData(&m_module, m_cubin));
        CU_SAFE_CALL(gpuDrvModuleGetFunction(&m_kernel, m_module, m_name.c_str()));

        m_loaded = true;

        delete[] m_cubin;
        m_cubin = nullptr;
    }

public:
    JitKernel(const JitKernel&) = delete;
    JitKernel(char *cubin, std::string_view name) : m_cubin{cubin}, m_name{name} {}

    void invoke(int num_blocks, int block_size, void **args,
                int shared_bytes) {
        ensure_loaded();

        if (shared_bytes > m_max_dynamic_shared_bytes) {
            CU_SAFE_CALL(gpuDrvFuncSetAttribute(
                m_kernel, gpuDrvFuncAttributeMaxDynamicSharedMemorySize,
                shared_bytes));
            m_max_dynamic_shared_bytes = shared_bytes;
        }

        CU_SAFE_CALL(gpuDrvLaunchKernel(m_kernel, num_blocks, 1, 1,
                                        block_size, 1, 1, shared_bytes,
                                        NULL, args, 0));

        CUDA_SAFE_CALLN(gpuPeekAtLastError());
        if (jit_debug) CUDA_SAFE_CALLN(gpuStreamSynchronize(0));
    }
};

struct ExecutionSection {
    int start;
    int end;

    int size() const { return end - start; }
};

class ExecutionSections {
private:
    enum class Kind {
        direct,
        atomics,
        color2,
    };

    Kind m_kind;
    op_set m_set = nullptr;
    op_plan *m_plan = nullptr;
    bool m_separate_owned = false;

    ExecutionSections(Kind kind, op_set set, op_plan *plan,
                      bool separate_owned)
        : m_kind{kind}, m_set{set}, m_plan{plan},
          m_separate_owned{separate_owned} {}

public:
    static ExecutionSections direct(op_set set) {
        return ExecutionSections{Kind::direct, set, nullptr, false};
    }

    static ExecutionSections atomics(op_set set, bool separate_owned) {
        return ExecutionSections{Kind::atomics, set, nullptr, separate_owned};
    }

    static ExecutionSections color2(op_plan *plan) {
        return ExecutionSections{Kind::color2, nullptr, plan, false};
    }

    int size() const {
        switch (m_kind) {
        case Kind::direct:
            return 1;
        case Kind::atomics:
            return m_separate_owned ? 3 : 2;
        case Kind::color2:
            return m_plan->ncolors;
        }

        assert(false);
        return 0;
    }

    ExecutionSection operator[](int index) const {
        assert(index >= 0 && index < size());

        switch (m_kind) {
        case Kind::direct:
            return {0, static_cast<int>(m_set->size)};
        case Kind::atomics:
            if (index == 0)
                return {0, m_set->core_size};
            if (m_separate_owned && index == 1)
                return {m_set->core_size, static_cast<int>(m_set->size)};

            return {m_separate_owned ? static_cast<int>(m_set->size)
                                     : m_set->core_size,
                    static_cast<int>(m_set->size) + m_set->exec_size};
        case Kind::color2:
            return {m_plan->col_offsets[0][index],
                    m_plan->col_offsets[0][index + 1]};
        }

        assert(false);
        return {0, 0};
    }
};

enum class KernelVariant {
    baseline,
    staged,
};

struct KernelExecutionOptions {
    KernelVariant variant = KernelVariant::baseline;
    int shared_bytes = 0;
};

struct KernelExecution {
    KernelVariant variant;
    JitKernel *jit_kernel;
    ExecutionSections sections;
    int block_size;
    int block_limit;
    int max_blocks;
    int shared_bytes;

    int num_blocks(int section_index) const {
        auto section = sections[section_index];
        int blocks = (section.size() + block_size - 1) / block_size;
        return std::min(blocks, block_limit);
    }
};

enum class ParamType {
    i32,
    i64,
    f32,
    f64,
    logical,
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

    uint64_t m_hash_last = 0;

    uint64_t m_hash_device = 0;
    uint64_t* m_hash_device_ptr = nullptr;

public:
    template<typename T>
    JitParam(std::string_view name, T *data, T *data_d = nullptr,
             uint64_t *hash_device_ptr = nullptr)
        : m_name{name}, m_data{data}, m_data_d{data_d}, m_n_elems{1}, m_elem_size{sizeof(T)},
          m_type{JitTypes<T>::value}, m_array{false}, m_hash_device_ptr{hash_device_ptr} {}

    template<typename T>
    JitParam(std::string_view name, T *data, std::size_t len, T *data_d = nullptr,
             uint64_t *hash_device_ptr = nullptr)
        : m_name{name}, m_data{data}, m_data_d{data_d}, m_n_elems{len}, m_elem_size{sizeof(T)},
          m_type{JitTypes<T>::value}, m_array{true}, m_hash_device_ptr{hash_device_ptr} {}

    uint64_t hash() {
        m_hash_last = op::f2c::hash(m_data, m_n_elems * m_elem_size, hash_seed_default);
        return m_hash_last;
    }

    void upload() {
        if (m_data_d == nullptr) return;

        auto hash_device = m_hash_device_ptr != nullptr ? *m_hash_device_ptr : m_hash_device;
        if (m_hash_last == hash_device) return;

        CUDA_SAFE_CALL(gpuMemcpyAsync(m_data_d, m_data, m_elem_size * m_n_elems,
                       gpuMemcpyHostToDevice));

        if (m_hash_device_ptr != nullptr)
            *m_hash_device_ptr = m_hash_last;
        else
            m_hash_device = m_hash_last;
    }

    std::string format_type() {
        switch (m_type) {
            case ParamType::i32:     return "int";
            case ParamType::i64:     return "int64_t";
            case ParamType::f32:     return "float";
            case ParamType::f64:     return "double";
            case ParamType::logical: return "bool";
        }

        __builtin_unreachable();
    }

    std::string format_value() {
        std::ostringstream os;
        if (m_array) os << "{ ";

        for (std::size_t i = 0; i < m_n_elems; ++i) {
            char *elem = (char *)m_data + m_elem_size * i;

            switch (m_type) {
                case ParamType::i32:     os << *((int *)elem); break;
                case ParamType::i64:     os << *((int64_t *)elem); break;
                case ParamType::f32:     os << std::hexfloat << *((float *)elem); break;
                case ParamType::f64:     os << std::hexfloat << *((double *)elem); break;
                case ParamType::logical: os << std::boolalpha << *((bool *)elem); break;
            }

            if (m_array && i < m_n_elems - 1) os << ", ";
        }

        if (m_array) os << " }";
        return os.str();
    }

    std::string format() {
        std::ostringstream os;

        os << "static constexpr " << format_type() << " " << m_name;
        if (m_array) { os << "[" << m_n_elems << "]"; }
        os << " = " << format_value() << ";" << std::endl;

        return os.str();
    }
};

struct HashInfo {
    std::size_t count = 0;
    bool jit_started = false;
    std::thread jit_thread;
};

struct KernelImplementation {
    std::string name;
    const void *offline_kernel;
    gpuFuncAttributes_t offline_attrs;
    std::string source;

    int offline_max_dynamic_shared_bytes = 0;
    std::unordered_map<uint64_t, JitKernel> jit_kernels;
    std::unordered_map<uint64_t, HashInfo> hash_infos;

    KernelImplementation(std::string_view name_, const void *offline_kernel_,
                         std::string_view source_)
        : name{name_}, offline_kernel{offline_kernel_}, source{source_} {}
};

class KernelInfo {
private:
    std::string m_name;
    KernelImplementation m_baseline;
    std::unique_ptr<KernelImplementation> m_staged;
    std::vector<JitParam> m_params;
    std::mutex m_jit_kernels_mutex;

    KernelImplementation& implementation(KernelVariant variant) {
        if (variant == KernelVariant::baseline)
            return m_baseline;

        if (m_staged == nullptr) {
            std::fprintf(stderr,
                         "error: staged implementation requested but not registered (in %s)\n",
                         m_name.c_str());
            std::exit(1);
        }

        return *m_staged;
    }

    bool is_jit_candidate(const KernelImplementation& impl) {
        return jit_force || impl.offline_attrs.numRegs > 32;
    }

    uint64_t hash_params() {
        uint64_t hash = hash_seed_default;

        for (auto& param : m_params)
            hash = op::f2c::hash(param.hash(), hash);

        return hash;
    }

    std::string format_params() {
        auto src = std::string();

        for (auto& param : m_params)
            src += param.format();

        return src;
    }

    std::thread compile(KernelImplementation& impl, uint64_t hash) {
        ++jit_active_threads;

        std::string jit_src = std::string("#include <op_f2c_prelude.h>\n") +
#ifdef OP_F2C_PARAMS
                              std::string("#include <op_f2c_params.h>\n") +
#endif
                              std::string("\nnamespace f2c = op::f2c;\n") +
                              format_params() + impl.source;
        
        // std::cout << "JIT source [" << impl.name << " (hash " << std::hex << hash << std::dec << ")]:" <<
        //     " ***\n" << jit_src << "\n***\n\n";
        
        auto do_compile = [this, &impl](auto jit_src, auto hash) {
#ifdef OP_F2C_PARAMS
            const char *headers[] = { OP_F2C_PRELUDE_DATA, OP_F2C_PARAMS_DATA };
            const char *header_names[] = { "op_f2c_prelude.h", "op_f2c_params.h" };
#else
            const char *headers[] = { OP_F2C_PRELUDE_DATA };
            const char *header_names[] = { "op_f2c_prelude.h" };
#endif
            gpuRtcProgram_t prog;
            NVRTC_SAFE_CALL(gpuRtcCreateProgram(&prog, jit_src.c_str(), impl.name.c_str(),
                            sizeof(headers) / sizeof(headers[0]), headers, header_names));

#ifdef OP2_CUDA
            const char *opts[] = {
                jit_arch.c_str(),
                "--std=c++20",
#if __CUDACC_VER_MAJOR__ >= 12 && __CUDACC_VER_MINOR__ >= 4
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
                    char *log = new char[log_size];
                    NVRTC_SAFE_CALL(gpuRtcGetProgramLog(prog, log));

                    std::printf("%s\n", log);
                    delete[] log;
                }

                exit(1);
            }

            size_t cubin_size;
            NVRTC_SAFE_CALL(gpuRtcGetCodeSize(prog, &cubin_size));

            char *cubin = new char[cubin_size];
            NVRTC_SAFE_CALL(gpuRtcGetCode(prog, cubin));
            NVRTC_SAFE_CALL(gpuRtcDestroyProgram(&prog));

            std::scoped_lock lock(m_jit_kernels_mutex);
            auto [it, inserted] = impl.jit_kernels.emplace(std::piecewise_construct,
                    std::forward_as_tuple(hash),
                    std::forward_as_tuple(cubin, impl.name));

            assert(inserted);
            --jit_active_threads;
        };

        std::thread compilation_thread(do_compile, jit_src, hash);
        return compilation_thread;
    }

    void invoke_offline(KernelImplementation& impl, int num_blocks,
                        int block_size, void **args,
                        int shared_bytes) {
        for (auto& param : m_params)
            param.upload();

        if (shared_bytes > impl.offline_max_dynamic_shared_bytes) {
            CUDA_SAFE_CALLN(gpuFuncSetAttribute(
                impl.offline_kernel,
                gpuFuncAttributeMaxDynamicSharedMemorySize,
                shared_bytes));
            impl.offline_max_dynamic_shared_bytes = shared_bytes;
        }

        CUDA_SAFE_CALLN(gpuLaunchKernel(impl.offline_kernel, num_blocks,
                                        block_size, args,
                                        shared_bytes, 0));
        CUDA_SAFE_CALLN(gpuPeekAtLastError());

        if (jit_debug) CUDA_SAFE_CALLN(gpuStreamSynchronize(0));
    }

    template<typename T>
    T *lookup_symbol(const T *symbol) {
        if (symbol == nullptr) return nullptr;

        T *data_d = nullptr;
        CUDA_SAFE_CALL(gpuGetSymbolAddress((void **)&data_d, (const void *)symbol));

        return data_d;
    }

public:
    KernelInfo(const KernelInfo&) = delete;
    KernelInfo(std::string_view name, const void *kernel, std::string_view src)
        : m_name{name}, m_baseline{name, kernel, src} {
        jit_init();
        CUDA_SAFE_CALL(gpuFuncGetAttributes(&m_baseline.offline_attrs,
                                            m_baseline.offline_kernel));
    }

    ~KernelInfo() {
        auto join_compilations = [](KernelImplementation& impl) {
            for (auto& [hash, hash_info] : impl.hash_infos) {
                if (hash_info.jit_thread.joinable())
                    hash_info.jit_thread.join();
            }
        };

        join_compilations(m_baseline);
        if (m_staged != nullptr)
            join_compilations(*m_staged);
    }

    void register_smem_variant(std::string_view name, const void *kernel,
                               std::string_view src) {
        if (m_staged != nullptr) {
            std::fprintf(stderr,
                         "error: staged implementation already registered (in %s)\n",
                         m_name.c_str());
            std::exit(1);
        }

        auto staged = std::make_unique<KernelImplementation>(name, kernel, src);
        CUDA_SAFE_CALL(gpuFuncGetAttributes(&staged->offline_attrs,
                                            staged->offline_kernel));
        m_staged = std::move(staged);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, const T *symbol = nullptr,
                   uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, lookup_symbol(symbol), hash_device_ptr);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, std::size_t len, const T *symbol = nullptr,
                   uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, len, lookup_symbol(symbol), hash_device_ptr);
    }

private:
    JitKernel *get_kernel(KernelImplementation& impl) {
        auto hash = hash_params();

        if (!jit_enable || !is_jit_candidate(impl))
            return nullptr;

        auto [hash_elem, inserted] = impl.hash_infos.insert({hash, HashInfo()});
        hash_elem->second.count++;

        {
            std::scoped_lock lock(m_jit_kernels_mutex);
            auto kernel_elem = impl.jit_kernels.find(hash);
            if (kernel_elem != impl.jit_kernels.end())
                return &kernel_elem->second;
        }

        if (hash_elem->second.count > 8 && !hash_elem->second.jit_started && jit_active_threads < jit_max_threads) {
            if (jit_debug)
                std::printf("compiling %s for hash %lx\n",
                            impl.name.c_str(), hash);

            hash_elem->second.jit_started = true;
            hash_elem->second.jit_thread = compile(impl, hash);

            if (jit_seq_compile)
                hash_elem->second.jit_thread.join();
        }

        return nullptr;
    }

    std::tuple<int, int> get_launch_config(JitKernel *kernel, int n_elems) {
        return {INT32_MAX, 128};
    }

public:
    template<typename PrepareParams>
    KernelExecution prepare(op_arg *args, int nargs, ExecutionSections sections,
                            KernelExecutionOptions options,
                            PrepareParams prepare_params) {
        auto& impl = implementation(options.variant);

        int max_section_size = 0;
        for (int i = 0; i < sections.size(); ++i)
            max_section_size = std::max(max_section_size, sections[i].size());

        auto [block_limit, block_size] = get_launch_config(nullptr, max_section_size);
        block_limit = std::min(block_limit,
                               ::getBlockLimit(args, nargs, block_size, m_name.c_str()));

        int max_blocks = 0;
        for (int i = 0; i < sections.size(); ++i) {
            int section_blocks = (sections[i].size() + block_size - 1) / block_size;
            max_blocks = std::max(max_blocks, section_blocks);
        }

        max_blocks = std::min(max_blocks, block_limit);

        KernelExecution execution{options.variant, nullptr, sections, block_size,
                                  block_limit, max_blocks, options.shared_bytes};

        // Some JIT parameters, such as the C++ backend's global-reduction
        // stride, depend on the launch dimensions. Set them before hashing.
        prepare_params(execution);
        execution.jit_kernel = get_kernel(impl);

        return execution;
    }

    template<typename PrepareParams>
    KernelExecution prepare(op_arg *args, int nargs, ExecutionSections sections,
                            PrepareParams prepare_params) {
        return prepare(args, nargs, sections, KernelExecutionOptions{},
                       prepare_params);
    }

    KernelExecution prepare(op_arg *args, int nargs, ExecutionSections sections) {
        return prepare(args, nargs, sections, KernelExecutionOptions{});
    }

    KernelExecution prepare(op_arg *args, int nargs, ExecutionSections sections,
                            KernelExecutionOptions options) {
        return prepare(args, nargs, sections, options,
                       [](const KernelExecution&) {});
    }

    void invoke(const KernelExecution& execution, int num_blocks, void **args,
                void **args_jit) {
        auto& impl = implementation(execution.variant);

        if (execution.jit_kernel == nullptr) {
            op_profile_next("Offline Kernel");
            invoke_offline(impl, num_blocks, execution.block_size, args,
                           execution.shared_bytes);

            return;
        }

        op_profile_next("JIT Kernel");
        execution.jit_kernel->invoke(num_blocks, execution.block_size, args_jit,
                                     execution.shared_bytes);
    }
};

} // namespace op::f2c
