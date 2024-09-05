#pragma once

#include <extern/rapidhash.h>
#include <op_timing2.h>

#include <nvrtc.h>
#include <cuda.h>

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


#define NVRTC_SAFE_CALL(x)                                                          \
    do {                                                                            \
        nvrtcResult result = x;                                                     \
        if (result != NVRTC_SUCCESS) {                                              \
            const char *msg = nvrtcGetErrorString(result);                          \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,         \
                    __FILE__, __LINE__);                                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

#define CU_SAFE_CALL(x)                                                             \
    do {                                                                            \
        CUresult result = x;                                                        \
        if (result != CUDA_SUCCESS) {                                               \
            const char *msg;                                                        \
            cuGetErrorName(result, &msg);                                           \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d (in %s)\n", msg, \
                    __FILE__, __LINE__, m_name.c_str());                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

#define CUDA_SAFE_CALLN(x)                                                          \
    do {                                                                            \
        cudaError_t result = x;                                                     \
        if (result != cudaSuccess) {                                                \
            const char *msg = cudaGetErrorString(result);                           \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d (in %s)\n", msg, \
                    __FILE__, __LINE__, m_name.c_str());                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

#define CUDA_SAFE_CALL(x)                                                           \
    do {                                                                            \
        cudaError_t result = x;                                                     \
        if (result != cudaSuccess) {                                                \
            const char *msg = cudaGetErrorString(result);                           \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,         \
                    __FILE__, __LINE__);                                            \
            exit(1);                                                                \
        }                                                                           \
    } while(0)


namespace op::f2c {

constexpr uint64_t hash_seed_default = RAPID_SEED;

static bool jit_initialized = false;

static bool jit_enable = true;
static bool jit_seq_compile = false;
static bool jit_debug = false;
static int jit_max_threads = INT32_MAX;

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
    CUDA_SAFE_CALL(cudaGetDevice(&device));

    cudaDeviceProp props;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, device));

    int cc = props.major * 10 + props.minor;
    jit_arch = "-arch=sm_" + std::to_string(cc);

    if (jit_debug)
        std::printf("JIT arch flag: %s\n", jit_arch.c_str());

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
    char *m_cubin;

    CUmodule m_module;
    CUfunction m_kernel;

    // std::vector<std::tuple<int, int>> m_launch_configs;

    void ensure_loaded() {
        if (m_loaded) return;

        CU_SAFE_CALL(cuModuleLoadData(&m_module, m_cubin));
        CU_SAFE_CALL(cuModuleGetFunction(&m_kernel, m_module, m_name.c_str()));

        m_loaded = true;

        delete[] m_cubin;
        m_cubin = nullptr;
    }

public:
    JitKernel(const JitKernel&) = delete;
    JitKernel(char *cubin, std::string_view name) : m_cubin{cubin}, m_name{name} {}

    /*
    const std::vector<std::tuple<int, int>>& get_launch_configs() {
        if (m_launch_configs.size() > 0) return m_launch_configs;

        ensure_loaded();

        int min_grid_size, block_size;
        CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, m_kernel,
                                                      nullptr, 0, INT32_MAX));

        m_launch_configs.push_back({min_grid_size, block_size});
        while (block_size > 32) {
            int next_block_size = (block_size & ~(-32)) ? block_size & (-32) : block_size - 32;
            CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, m_kernel,
                                                          nullptr, 0, next_block_size));

            m_launch_configs.push_back({min_grid_size, block_size});
        }

        return m_launch_configs;
    }
    */

    void invoke(int num_blocks, int block_size, void **args) {
        ensure_loaded();
        CU_SAFE_CALL(cuLaunchKernel(m_kernel, num_blocks, 1, 1, block_size, 1, 1, 0,
                                    NULL, args, 0));

        CUDA_SAFE_CALLN(cudaPeekAtLastError());
        if (jit_debug) CUDA_SAFE_CALLN(cudaStreamSynchronize(0));
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

        CUDA_SAFE_CALL(cudaMemcpyAsync(m_data_d, m_data, m_elem_size * m_n_elems,
                    cudaMemcpyHostToDevice));

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

class KernelInfo {
private:
    std::string m_name;
    const void* m_kernel;

    cudaFuncAttributes m_kernel_attrs;
    // std::vector<std::tuple<int, int>> m_launch_configs;

    std::vector<JitParam> m_params;
    std::string m_src;

    std::mutex m_jit_kernels_mutex;
    std::unordered_map<uint64_t, JitKernel> m_jit_kernels;

    std::unordered_map<uint64_t, HashInfo> m_hash_infos;

    bool is_jit_candidate() {
        return m_kernel_attrs.numRegs > 32;
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

    std::thread compile(uint64_t hash) {
        ++jit_active_threads;

        std::string jit_src = std::string("#include <op_f2c_prelude.h>\n") +
                              std::string("#include <op_f2c_params.h>\n") +
                              std::string("\nnamespace f2c = op::f2c;\n") +
                              format_params() + m_src;

        auto do_compile = [&](auto jit_src, auto hash) {
            const char *headers[] = { op_f2c_prelude_data, op_f2c_params_data };
            const char *header_names[] = { "op_f2c_prelude.h", "op_f2c_params.h" };

            nvrtcProgram prog;
            NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, jit_src.c_str(), m_name.c_str(),
                                               2, headers, header_names));

            const char *opts[] = {
                jit_arch.c_str(),
                "--std=c++20",
                "--minimal",
                "--device-as-default-execution-space"
            };

            auto success = nvrtcCompileProgram(prog, 4, opts);
            if (success != NVRTC_SUCCESS) {
                size_t log_size;
                NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));

                if (log_size > 1) {
                    char *log = new char[log_size];
                    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));

                    std::printf("%s\n", log);
                    delete[] log;
                }

                exit(1);
            }

            size_t cubin_size;
            NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &cubin_size));

            char *cubin = new char[cubin_size];
            NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, cubin));
            NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

            std::scoped_lock lock(m_jit_kernels_mutex);
            auto [it, inserted] = m_jit_kernels.emplace(std::piecewise_construct,
                    std::forward_as_tuple(hash), std::forward_as_tuple(cubin, m_name));

            assert(inserted);
            --jit_active_threads;
        };

        std::thread compilation_thread(do_compile, jit_src, hash);
        return compilation_thread;
    }

    void invoke_offline(int num_blocks, int block_size, void **args) {
        for (auto& param : m_params)
            param.upload();

        CUDA_SAFE_CALLN(cudaLaunchKernel(m_kernel, num_blocks, block_size, args, 0, 0));
        CUDA_SAFE_CALLN(cudaPeekAtLastError());

        if (jit_debug) CUDA_SAFE_CALLN(cudaStreamSynchronize(0));
    }

    template<typename T>
    T *lookup_symbol(const T *symbol) {
        if (symbol == nullptr) return nullptr;

        T *data_d = nullptr;
        CUDA_SAFE_CALL(cudaGetSymbolAddress((void **)&data_d, (const void *)symbol));

        return data_d;
    }

public:
    KernelInfo(const KernelInfo&) = delete;
    KernelInfo(std::string_view name, const void *kernel, std::string_view src)
        : m_name{name}, m_kernel{kernel}, m_src{src} {
        jit_init();
        CUDA_SAFE_CALL(cudaFuncGetAttributes(&m_kernel_attrs, m_kernel));

        /*
        int min_grid_size, block_size;
        CUDA_SAFE_CALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, m_kernel));

        m_launch_configs.push_back({min_grid_size, block_size});
        while (block_size > 32) {
            int next_block_size = (block_size & ~(-32)) ? block_size & (-32) : block_size - 32;
            CUDA_SAFE_CALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, m_kernel,
                                                              0, next_block_size));

            m_launch_configs.push_back({min_grid_size, block_size});
        }
        */
    }

    ~KernelInfo() {
        for (auto& [hash, hash_info] : m_hash_infos) {
            if (hash_info.jit_thread.joinable())
                hash_info.jit_thread.join();
        }
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

    JitKernel *get_kernel() {
        auto hash = hash_params();

        if (!jit_enable || !is_jit_candidate())
            return nullptr;

        auto [hash_elem, inserted] = m_hash_infos.insert({hash, HashInfo()});
        hash_elem->second.count++;

        m_jit_kernels_mutex.lock();

        auto kernel_elem = m_jit_kernels.find(hash);
        if (kernel_elem != m_jit_kernels.end()) {
            m_jit_kernels_mutex.unlock();
            return &kernel_elem->second;
        }

        m_jit_kernels_mutex.unlock();

        if (hash_elem->second.count > 8 && !hash_elem->second.jit_started && jit_active_threads < jit_max_threads) {
            if (jit_debug) std::printf("compiling %s for hash %lx\n", m_name.c_str(), hash);

            hash_elem->second.jit_started = true;
            hash_elem->second.jit_thread = compile(hash);

            if (jit_seq_compile)
                hash_elem->second.jit_thread.join();
        }

        return nullptr;
    }

    std::tuple<int, int> get_launch_config(JitKernel *kernel, int n_elems) {
/*
        auto& launch_configs = kernel == nullptr ? m_launch_configs : kernel->get_launch_configs();

        int selected_config = 0;
        for (; selected_config < launch_configs.size() - 1; ++selected_config) {
            auto [min_grid_size, block_size] = launch_configs[selected_config];
            if (min_grid_size * block_size < n_elems) break;
        }
*/

        return {INT32_MAX, 64};
    }

    void invoke(JitKernel *kernel, int num_blocks, int block_size, void **args, void **args_jit) {
        if (kernel == nullptr) {
            op_timing2_next("Offline Kernel");
            invoke_offline(num_blocks, block_size, args);

            return;
        }

        op_timing2_next("JIT Kernel");
        kernel->invoke(num_blocks, block_size, args_jit);
    }
};

} // namespace op::f2c
