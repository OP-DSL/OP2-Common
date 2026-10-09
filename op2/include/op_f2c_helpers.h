#pragma once

#include <extern/rapidhash.h>
#include <op_lib_cpp.h>
#include <op_profile.h>
#include <op_gpu_shims.h>
#include <op_f2c_dispatch.h>
#include <op_hier_plan_cache.h>
#include <op_hier_plan.h>

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
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <utility>
// #include <iostream>

extern "C" {
int getBlockLimitWithPolicy(op_arg *args, int nargs, int block_size,
                            const char *name, bool gbl_inc_atomic);
void prepareDeviceGblsWithPolicy(op_arg *args, int nargs, int max_threads,
                                 bool gbl_inc_atomic);
bool processDeviceGblsWithPolicy(op_arg *args, int nargs, int nelems,
                                 int max_threads, bool gbl_inc_atomic);
op_plan *op_plan_get_stage(char const *name, op_set set, int part_size,
                           int nargs, op_arg *args, int ninds, int *inds,
                           int staging);
}

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

    init_strategy_config();

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

enum class ParamType {
    i32,
    i64,
    f32,
    f64,
    logical,
};

enum class ParamSource {
    external,
    scalar_arg,
    dat_stride,
    global_stride,
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
    uint64_t* m_hash_device_ptr = nullptr;

public:
    template<typename T>
    JitParam(std::string_view name, T *data, T *data_d = nullptr,
             uint64_t *hash_device_ptr = nullptr,
             ParamSource source = ParamSource::external,
             int arg_index = -1)
        : m_name{name}, m_data{data}, m_data_d{data_d}, m_n_elems{1}, m_elem_size{sizeof(T)},
          m_type{JitTypes<T>::value}, m_array{false}, m_source{source},
          m_arg_index{arg_index}, m_hash_device_ptr{hash_device_ptr} {}

    template<typename T>
    JitParam(std::string_view name, T *data, std::size_t len, T *data_d = nullptr,
             uint64_t *hash_device_ptr = nullptr)
        : m_name{name}, m_data{data}, m_data_d{data_d}, m_n_elems{len}, m_elem_size{sizeof(T)},
          m_type{JitTypes<T>::value}, m_array{true},
          m_source{ParamSource::external}, m_arg_index{-1},
          m_hash_device_ptr{hash_device_ptr} {}

    void update(op_arg *args, int nargs, const KernelExecution& execution) {
        switch (m_source) {
        case ParamSource::external:
            return;
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
        case ParamSource::global_stride:
            assert(m_elem_size == sizeof(int) && m_n_elems == 1);
            *static_cast<int *>(m_data) =
                execution.block_size * execution.max_blocks;
            return;
        }

        __builtin_unreachable();
    }

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

    // The argument groups a hierarchical strategy plans over.
    std::optional<HierArgGroups> groups;

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
    std::string m_profile_name;
    std::string m_profile_target;
    std::string m_profile_variant;
    LoopDescription m_loop;
    std::array<std::unique_ptr<KernelImplementation>, strategy_count>
        m_variants;
    std::array<detail::HierPlanCache, strategy_count> m_plan_caches;
    std::set<std::pair<Strategy, std::array<FallbackReason, strategy_count>>>
        m_reported;
    std::array<std::optional<std::size_t>, strategy_count> m_hier_capacity;
    bool m_plan_owner_registered = false;
    std::vector<JitParam> m_params;
    std::mutex m_jit_kernels_mutex;

    // Return a strategy's registered variant, or null.
    KernelImplementation *variant(Strategy strategy) {
        return m_variants[strategy_index(strategy)].get();
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

    // Wait for every JIT compile still running.
    void join_compilations() {
        for (auto& impl : m_variants) {
            if (impl == nullptr)
                continue;

            for (auto& [hash, hash_info] : impl->hash_infos) {
                if (hash_info.jit_thread.joinable())
                    hash_info.jit_thread.join();
            }
        }
    }

    // At op_exit, while NVRTC and the device are still up: a compile left
    // running until static destruction can crash or hang the exit.
    static void release_hier_plans_callback(void *owner) {
        auto *info = static_cast<KernelInfo *>(owner);
        info->join_compilations();
        for (auto& cache : info->m_plan_caches)
            cache.clear();
    }

    void register_plan_owner() {
        if (m_plan_owner_registered)
            return;

        register_hier_plan_owner(
            this, &KernelInfo::release_hier_plans_callback);
        m_plan_owner_registered = true;
    }

public:
    KernelInfo(const KernelInfo&) = delete;
    KernelInfo(std::string_view profile_name, std::string_view profile_target,
               std::string_view profile_variant, LoopDescription loop)
        : m_name{profile_name}, m_profile_name{profile_name},
          m_profile_target{profile_target}, m_profile_variant{profile_variant},
          m_loop{std::move(loop)} {
        jit_init();
        register_plan_owner();
    }

    ~KernelInfo() {
        join_compilations();

        if (m_plan_owner_registered)
            unregister_hier_plan_owner(this);
        for (auto& cache : m_plan_caches)
            cache.clear();
    }

    // Register the wrapper for one strategy; hierarchical strategies pass the
    // argument groups their plan is built over.
    void register_variant(Strategy strategy, std::string_view name,
                          const void *kernel, std::string_view src,
                          std::optional<HierArgGroups> groups = std::nullopt) {
        auto& slot = m_variants[strategy_index(strategy)];
        if (slot != nullptr) {
            std::fprintf(stderr,
                         "error: %s variant already registered (in %s)\n",
                         strategy_name(strategy).data(), m_name.c_str());
            std::exit(1);
        }

        assert(!groups.has_value() || hierarchical(strategy));
        auto impl = std::make_unique<KernelImplementation>(name, kernel, src);
        impl->groups = groups;
        CUDA_SAFE_CALL(gpuFuncGetAttributes(&impl->offline_attrs,
                                            impl->offline_kernel));
        slot = std::move(impl);
    }

    HierPlanCacheStatistics hier_plan_cache_statistics(Strategy strategy) const {
        return m_plan_caches[strategy_index(strategy)].statistics();
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

    template<typename T>
    void add_scalar_arg_param(std::string_view name, T *data, int arg_index,
                              const T *symbol = nullptr,
                              uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, lookup_symbol(symbol), hash_device_ptr,
                              ParamSource::scalar_arg, arg_index);
    }

    void add_dat_stride_param(std::string_view name, int *data, int arg_index,
                              const int *symbol = nullptr,
                              uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, lookup_symbol(symbol), hash_device_ptr,
                              ParamSource::dat_stride, arg_index);
    }

    void add_global_stride_param(std::string_view name, int *data,
                                 const int *symbol = nullptr,
                                 uint64_t *hash_device_ptr = nullptr) {
        m_params.emplace_back(name, data, lookup_symbol(symbol), hash_device_ptr,
                              ParamSource::global_stride, -1);
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

    // Return a hierarchical wrapper's usable dynamic shared-memory capacity.
    // OP2 fixes the device at initialization, so this is resolved once.
    std::size_t hier_capacity(Strategy strategy) {
        const auto *impl = variant(strategy);
        assert(impl != nullptr);
        auto& capacity = m_hier_capacity[strategy_index(strategy)];
        if (capacity.has_value())
            return *capacity;

        int device = -1;
        CUDA_SAFE_CALL(gpuGetDevice(&device));

        gpuDeviceProp_t properties;
        CUDA_SAFE_CALL(gpuGetDeviceProperties(&properties, device));

        std::size_t total = properties.sharedMemPerBlock;
#ifdef OP2_CUDA
        total = std::max(total,
                         static_cast<std::size_t>(
                             properties.sharedMemPerBlockOptin));
#endif
        auto static_bytes = static_cast<std::size_t>(
            impl->offline_attrs.sharedSizeBytes);
        capacity = total > static_bytes ? total - static_bytes : 0;
        return *capacity;
    }

    // Reject groups that omit an active indirect update the strategy must
    // see: increments, and for colouring read-writes as well.
    static bool all_indirect_updates_covered(
        std::span<const op_arg> args,
        const HierArgGroups& groups, bool read_write) {
        std::vector<bool> covered(args.size(), false);
        for (const auto& arg : groups.args) {
            assert(arg.arg_index >= 0 &&
                   static_cast<std::size_t>(arg.arg_index) < args.size());
            covered[static_cast<std::size_t>(arg.arg_index)] = true;
        }

        for (std::size_t i = 0; i < args.size(); ++i) {
            const op_arg& arg = args[i];
            bool update = arg.acc == OP_INC || (read_write && arg.acc == OP_RW);
            if (arg.opt != 0 && arg.argtype == OP_ARG_DAT && update &&
                arg.idx >= 0 && !covered[i])
                return false;
        }

        return true;
    }

    // Look up or build a hierarchical strategy's plan for the current loop
    // configuration.
    const detail::HierPlanCacheEntry& get_hier_plan(
        Strategy strategy, op_set set, std::span<const op_arg> args,
        std::span<const ExecutionSection> sections, int block_size) {
        assert(hierarchical(strategy));
        const HierArgGroups& groups = *variant(strategy)->groups;

        bool atomics = strategy == Strategy::hier_atomics;
        int chunk_size = strategy_config.chunk_size(strategy);
        HierPlanOptions plan_options{
            block_size, chunk_size > 0 ? chunk_size : OP_part_size,
            hier_capacity(strategy),
            atomics && strategy_config.hier_atomics_exclusive};
        auto key = detail::make_hier_plan_key(
            set, args, static_cast<int>(sections.size()), groups, plan_options);

        auto report = [&](const HierPlan& plan, bool rejected) {
            if (OP_diags <= 3)
                return;

            const auto& stats = plan.statistics;
            if (plan.colouring) {
                std::printf("hier_colouring: %s chunk %d, %zu chunks, "
                            "%zu launches, up to %d thread colours, "
                            "compression %.2fx\n",
                            m_profile_name.c_str(), plan.selected_chunk_size,
                            plan.num_chunks(), stats.launches,
                            stats.max_thread_colours, stats.compression());
                return;
            }

            std::printf(
                "hier_atomics: %s chunk %d, %zu chunks, %zu refs, "
                "%zu owners (%zu exclusive, %zu atomic), "
                "compression %.2fx%s\n",
                m_profile_name.c_str(), plan.selected_chunk_size,
                plan.num_chunks(), stats.raw_references,
                stats.distinct_targets, stats.exclusive_owners,
                stats.distinct_targets - stats.exclusive_owners,
                stats.compression(), rejected ? ", below the minimum" : "");
        };

        // The builder runs only for a new entry, so each plan reports once.
        auto& cache = m_plan_caches[strategy_index(strategy)];
        return cache.get_or_build(std::move(key), [&]() {
            if (!all_indirect_updates_covered(args, groups, !atomics))
                return HierPlanBuildResult{
                    FallbackReason::incompatible_argument, std::nullopt};

            if (!atomics) {
                auto result = build_hier_colouring_plan(
                    set, args, sections, groups, plan_options);
                if (result)
                    report(*result.plan, false);
                return result;
            }

            auto result = build_hier_atomics_plan(
                set, args, sections, groups, plan_options);
            if (!result)
                return result;

            // Drop a plan that barely combines references before it is
            // uploaded.
            bool rejected = result.plan->statistics.compression() <
                            strategy_config.hier_atomics_min_compression;
            report(*result.plan, rejected);
            if (rejected)
                return HierPlanBuildResult{FallbackReason::low_compression,
                                           std::nullopt};

            return result;
        });
    }

    // Report each distinct ladder outcome once per loop, at diagnostic
    // verbosity: the strategy that ran, and why each one above it did not.
    void report_selection(
        Strategy selected,
        const std::array<FallbackReason, strategy_count>& skipped) {
        if (OP_diags <= 3 || !m_reported.insert({selected, skipped}).second)
            return;

        std::string passed;
        for (Strategy strategy : strategies) {
            auto reason = skipped[strategy_index(strategy)];
            if (reason == FallbackReason::none)
                continue;

            passed += passed.empty() ? " (" : ", ";
            passed += strategy_name(strategy);
            passed += ": ";
            passed += fallback_reason_name(reason);
        }
        if (!passed.empty())
            passed += ")";

        std::printf("strategy: %s runs %s%s\n", m_profile_name.c_str(),
                    strategy_name(selected).data(), passed.c_str());
    }

    // The schedule every strategy but colouring runs: one section for a
    // direct loop, else core, owned (with a global reduction) and exec.
    ExecutionSchedule section_schedule(op_set set, bool global_reduction) const {
        if (m_loop.is_direct())
            return ExecutionSchedule::direct(set);

        return ExecutionSchedule::atomics(set, global_reduction);
    }

    // The colouring schedule, from op_plan.
    ExecutionSchedule colouring_schedule(op_set set, op_arg *args, int nargs) {
        op_profile_next("Plan");

        int part_size = m_loop.part_size() >= 0 ? m_loop.part_size()
                                                : OP_part_size;
        if (m_loop.nargs() != static_cast<std::size_t>(nargs) ||
            m_loop.ninds() == 0) {
            std::fprintf(stderr,
                         "error: invalid colouring indirect dat mapping (in %s)\n",
                         m_name.c_str());
            std::exit(1);
        }

        op_plan *plan = op_plan_get_stage(
            m_profile_name.c_str(), set, part_size, nargs, args,
            m_loop.ninds(), m_loop.indirect_dats(), OP_COLOR2);

        op_profile_next("Get Kernel");
        return ExecutionSchedule::colouring(set, plan);
    }

    // Resolve the physical launch policy for a schedule.
    std::pair<int, int> launch_config(const ExecutionSchedule& schedule,
                                      op_arg *args, int nargs) {
        int max_section_size = 0;
        for (int i = 0; i < schedule.size(); ++i)
            max_section_size = std::max(max_section_size, schedule[i].size());

        auto [block_limit, block_size] = get_launch_config(nullptr, max_section_size);
        block_limit = std::min(
            block_limit,
            ::getBlockLimitWithPolicy(args, nargs, block_size, m_name.c_str(),
                                      m_loop.gbl_inc_atomic()));
        return {block_limit, block_size};
    }

    // Walk the ladder: run the first registered strategy that is enabled and,
    // for a hierarchical one, whose plan accepts this configuration.  When
    // none does, the lowest non-hierarchical strategy runs anyway, keeping
    // its own reason.
    KernelExecution prepare(op_set set, op_arg *args, int nargs,
                            bool global_reduction,
                            const KernelExecutionOptions& options) {
        ExecutionSchedule schedule = section_schedule(set, global_reduction);
        int block_limit = 0;
        int block_size = 0;
        std::tie(block_limit, block_size) = launch_config(schedule, args, nargs);

        std::array<ExecutionSection, 3> sections;
        assert(schedule.size() <= static_cast<int>(sections.size()));
        for (int i = 0; i < schedule.size(); ++i)
            sections[static_cast<std::size_t>(i)] = schedule[i];

        std::array<FallbackReason, strategy_count> skipped{};
        std::optional<Strategy> selected;
        std::optional<Strategy> floor;
        const detail::HierPlanCacheEntry *entry = nullptr;
        bool started = !options.force.has_value();

        for (Strategy strategy : strategies) {
            KernelImplementation *impl = variant(strategy);
            if (impl == nullptr)
                continue;
            if (!hierarchical(strategy))
                floor = strategy;

            started |= options.force == strategy;
            if (!started || selected.has_value())
                continue;

            bool forced = options.force == strategy;
            auto& reason = skipped[strategy_index(strategy)];
            if (!forced && !strategy_config.enabled(strategy))
                reason = FallbackReason::disabled;

            if (reason == FallbackReason::none && impl->groups.has_value()) {
                const auto& cached = get_hier_plan(
                    strategy, set,
                    std::span<const op_arg>{args,
                                            static_cast<std::size_t>(nargs)},
                    std::span<const ExecutionSection>{
                        sections.data(),
                        static_cast<std::size_t>(schedule.size())},
                    block_size);
                reason = cached.reason();
                if (cached)
                    entry = &cached;
            }

            if (reason == FallbackReason::none)
                selected = strategy;
        }

        assert(selected.has_value() || floor.has_value());
        Strategy strategy = selected.value_or(*floor);

        const HierPlan *plan = nullptr;
        HierPlanDeviceView plan_device;
        if (entry != nullptr) {
            plan = entry->plan();
            plan_device = entry->device_view();

            // The wrapper's single stride argument indexes both the maps and
            // the plan's per-element arrays, so the plan has to have been
            // built against the same stride the schedule reports.
            assert(plan->set_stride == schedule.set_stride());
        }

        if (strategy == Strategy::colouring) {
            schedule = colouring_schedule(set, args, nargs);
            std::tie(block_limit, block_size) =
                launch_config(schedule, args, nargs);
        }

        KernelExecution execution{
            strategy, nullptr, schedule, block_size, block_limit, 0,
            options.shared_bytes, plan, plan_device, skipped};

        // Reduction scratch uses the selected plan's capped physical grid.
        for (int i = 0; i < schedule.size(); ++i)
            for (int launch = 0; launch < execution.launches(i); ++launch)
                execution.max_blocks = std::max(
                    execution.max_blocks, execution.num_blocks(i, launch));

        for (auto& param : m_params)
            param.update(args, nargs, execution);

        report_selection(strategy, skipped);
        execution.jit_kernel = get_kernel(*variant(strategy));

        return execution;
    }

    static bool has_global_reduction(op_arg *args, int nargs) {
        for (int i = 0; i < nargs; ++i) {
            if (args[i].opt == 0 || args[i].argtype != OP_ARG_GBL)
                continue;

            if (args[i].acc == OP_INC || args[i].acc == OP_MIN ||
                args[i].acc == OP_MAX)
                return true;
        }

        return false;
    }

    static bool has_global_output(op_arg *args, int nargs) {
        for (int i = 0; i < nargs; ++i) {
            if (args[i].opt == 0 || args[i].argtype != OP_ARG_GBL)
                continue;

            if (args[i].acc == OP_INC || args[i].acc == OP_MIN ||
                args[i].acc == OP_MAX || args[i].acc == OP_RW ||
                args[i].acc == OP_WRITE)
                return true;
        }

        return false;
    }

    static bool is_type(const char *actual, const char *expected) {
        return actual != nullptr && std::strcmp(actual, expected) == 0;
    }

    void reduce_mpi_globals(op_arg *args, int nargs) {
        for (int i = 0; i < nargs; ++i) {
            auto& arg = args[i];
            if (arg.opt == 0 || arg.argtype != OP_ARG_GBL ||
                (arg.acc != OP_INC && arg.acc != OP_MIN && arg.acc != OP_MAX))
                continue;

            if (is_type(arg.type, "double") || is_type(arg.type, "r8") ||
                is_type(arg.type, "real*8") || is_type(arg.type, "real(8)")) {
                op_mpi_reduce_double(&arg, reinterpret_cast<double *>(arg.data));
            } else if (is_type(arg.type, "float") || is_type(arg.type, "r4") ||
                       is_type(arg.type, "real*4") || is_type(arg.type, "real(4)")) {
                op_mpi_reduce_float(&arg, reinterpret_cast<float *>(arg.data));
            } else if (is_type(arg.type, "int") || is_type(arg.type, "i4") ||
                       is_type(arg.type, "integer*4") ||
                       is_type(arg.type, "integer(4)")) {
                op_mpi_reduce_int(&arg, reinterpret_cast<int *>(arg.data));
            } else if (is_type(arg.type, "bool") ||
                       is_type(arg.type, "logical")) {
                op_mpi_reduce_bool(&arg, reinterpret_cast<bool *>(arg.data));
            } else {
                std::fprintf(stderr,
                             "error: unsupported MPI reduction type '%s' (in %s)\n",
                             arg.type == nullptr ? "<null>" : arg.type,
                             m_name.c_str());
                std::exit(1);
            }
        }
    }

    void launch_section(const KernelExecution& execution, int section_index,
                        int num_blocks,
                        void **args, void **args_jit) {
        auto& impl = *variant(execution.strategy);

        if (execution.jit_kernel == nullptr) {
            op_profile_next("Offline Kernel");
            invoke_offline(impl, num_blocks, execution.block_size, args,
                           execution.dynamic_shared_bytes(section_index));

            return;
        }

        op_profile_next("JIT Kernel");
        execution.jit_kernel->invoke(num_blocks, execution.block_size, args_jit,
                                     execution.dynamic_shared_bytes(
                                         section_index));
    }

    template<Strategy S, typename Bindings>
    void bind_and_launch(const KernelExecution& execution, int section_index,
                         int num_blocks,
                         LaunchContext& launch, op_arg *args,
                         Bindings& bindings) {
        auto kernel_args = bindings.template make_arguments<S>(launch, args);
        launch_section(execution, section_index, num_blocks,
                       kernel_args.offline.data(),
                       kernel_args.jit.data());
    }

    // Call f.template operator()<S>() for the strategy S, among those the
    // bindings generate arguments for, that the execution selected.
    template<typename Bindings, typename F>
    static void with_strategy(Strategy strategy, F&& f) {
        [[maybe_unused]] bool found = [&]<std::size_t... I>(
                                          std::index_sequence<I...>) {
            return ((strategy == Bindings::strategies[I]
                         ? (f.template operator()<Bindings::strategies[I]>(),
                            true)
                         : false) ||
                    ...);
        }(std::make_index_sequence<Bindings::strategies.size()>{});
        assert(found);
    }

public:
    template<typename Bindings>
    KernelInvocationResult invoke(
        op_set set, op_arg *args, int nargs, Bindings bindings,
        KernelExecutionOptions options = KernelExecutionOptions{}) {
        op_profile_enter_kernel(m_profile_name.c_str(), m_profile_target.c_str(),
                                m_profile_variant.c_str());
        op_profile_enter("Init");
        op_profile_enter("MPI Exchanges");
        int n_exec = op_mpi_halo_exchanges(set, nargs, args, 2);

        if (n_exec == 0) {
            op_profile_exit();
            op_profile_exit();

            op_mpi_wait_all(nargs, args);
            reduce_mpi_globals(args, nargs);
            op_mpi_set_dirtybit_cuda(nargs, args);
            op_profile_exit();

            return {};
        }

        bool global_reduction = has_global_reduction(args, nargs);
        bool global_output = has_global_output(args, nargs);

        op_profile_next("Get Kernel");
        auto execution = prepare(set, args, nargs, global_reduction, options);
        const auto& schedule = execution.schedule;
        op_profile_exit();

        op_profile_enter("Prepare GBLs");
        int global_stride = execution.block_size * execution.max_blocks;
        prepareDeviceGblsWithPolicy(args, nargs, global_stride,
                                    m_loop.gbl_inc_atomic());

        GlobalInitContext global_init{execution.block_size,
                                      execution.max_blocks, global_stride};
        bindings.init_globals(global_init, args);
        op_profile_exit();

        op_profile_next("Computation");
        op_profile_enter("Kernel");

        bool exit_sync = false;
        for (int section_index = 0; section_index < schedule.size();
             ++section_index) {
            if (schedule.wait_before(section_index)) {
                op_profile_next("MPI Wait");
                op_mpi_wait_all(nargs, args);
                op_profile_next("Kernel");
            }

            auto section = schedule[section_index];
            for (int launch_index = 0;
                 section.size() > 0 &&
                 launch_index < execution.launches(section_index);
                 ++launch_index) {
                LaunchContext launch{
                    global_stride,
                    0,
                    section.start,
                    section.end,
                    schedule.set_stride(),
                    schedule.color_reorder(),
                    {}};

                if (execution.plan != nullptr) {
                    launch.hier.plan = execution.plan_device;
                    std::tie(launch.hier.chunk_begin, launch.hier.chunk_end) =
                        execution.chunk_range(section_index, launch_index);
                    if (const auto& staging = execution.plan->staging)
                        launch.hier.has_exclusive = staging->has_exclusive;
                }

                int num_blocks =
                    execution.num_blocks(section_index, launch_index);
                with_strategy<Bindings>(
                    execution.strategy, [&]<Strategy S>() {
                        bind_and_launch<S>(execution, section_index,
                                           num_blocks, launch, args,
                                           bindings);
                    });
            }

            if (global_output &&
                schedule.process_globals_after(section_index)) {
                op_profile_next("Process GBLs");
                exit_sync |= processDeviceGblsWithPolicy(
                    args, nargs, global_stride, global_stride,
                    m_loop.gbl_inc_atomic());
                op_profile_next("Kernel");
            }
        }

        if (schedule.wait_before(schedule.size())) {
            op_profile_next("MPI Wait");
            op_mpi_wait_all(nargs, args);
            op_profile_next("Kernel");
        }

        op_profile_exit();
        op_profile_exit();

        op_profile_enter("Finalise");
        if (exit_sync)
            CUDA_SAFE_CALL(gpuStreamSynchronize(0));
        reduce_mpi_globals(args, nargs);
        op_mpi_set_dirtybit_cuda(nargs, args);

        op_profile_exit();
        op_profile_exit();

        return {execution.jit_kernel != nullptr, execution.strategy,
                execution.block_size, execution.max_blocks, execution.skipped};
    }
};

} // namespace op::f2c
