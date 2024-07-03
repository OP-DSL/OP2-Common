#pragma once

#include <extern/rapidhash.h>
#include <op_timing2.h>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#include <array>
#include <vector>
#include <unordered_map>
#include <string>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <mutex>


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
                    __FILE__, __LINE__, name.c_str());                              \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

#define CUDA_SAFE_CALLN(x)                                                          \
    do {                                                                            \
        cudaError_t result = x;                                                     \
        if (result != cudaSuccess) {                                                \
            const char *msg = cudaGetErrorString(result);                           \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d (in %s)\n", msg, \
                    __FILE__, __LINE__, name.c_str());                              \
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

    char *seq_compile_str = std::getenv("OP_JIT_SEQ_COMPILE");
    if (seq_compile_str != nullptr) {
        auto seq_compile = std::string(seq_compile_str);
        std::transform(seq_compile.begin(), seq_compile.end(), seq_compile.begin(),
            [](auto c){ return std::tolower(c); });

        if (seq_compile == "1" || seq_compile == "yes" || seq_compile == "true")
            jit_seq_compile = true;
    }

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

struct jit_kernel {
    std::string name;

    bool loaded = false;
    char *cubin;

    CUmodule module;
    CUfunction kernel;

    bool profile = false;

    jit_kernel(const jit_kernel&) = delete;

    jit_kernel(jit_kernel&& other) : cubin{other.cubin}, module{other.module}, kernel{other.kernel} {
        other.cubin = nullptr;
        other.module = nullptr;
        other.kernel = nullptr;
    };

    jit_kernel(char *cubin, std::string_view name) : cubin{cubin}, name{name} {
        // profile = name == "op2_k_flux_2_vflux_edge_main_wrapper";
        profile = name == "op2_k_flux_18_grad_edge_work_1_main_wrapper";
    }

    /*
    ~jit_kernel() {
        CU_SAFE_CALL(cuModuleUnload(module));
    }
    */

    void invoke(int num_blocks, int block_size, void **args) {
        if (!loaded) {
            CU_SAFE_CALL(cuModuleLoadData(&module, cubin));
            CU_SAFE_CALL(cuModuleGetFunction(&kernel, module, name.c_str()));

            loaded = true;

            delete[] cubin;
            cubin = nullptr;
        }

        if (profile) CUDA_SAFE_CALL(cudaProfilerStart());
        CU_SAFE_CALL(cuLaunchKernel(kernel, num_blocks, 1, 1, block_size, 1, 1, 0,
                                      NULL, args, 0));
        if (profile) CUDA_SAFE_CALL(cudaProfilerStop());

        CUDA_SAFE_CALLN(cudaPeekAtLastError());
        // CUDA_SAFE_CALLN(cudaStreamSynchronize(0));
    }
};

enum class param_type {
    i32,
    i64,
    f32,
    f64,
    logical,
};

template<typename T> struct jit_types {};

template<> struct jit_types<int>      { static const param_type value = param_type::i32; };
template<> struct jit_types<int64_t>  { static const param_type value = param_type::i64; };
template<> struct jit_types<float>    { static const param_type value = param_type::f32; };
template<> struct jit_types<double>   { static const param_type value = param_type::f64; };
template<> struct jit_types<bool>     { static const param_type value = param_type::logical; };

struct jit_param {
    std::string name;

    void *data;

    std::size_t n_elems;
    std::size_t elem_size;

    param_type type;
    bool array;

    template<typename T>
    jit_param(std::string_view name, T *data)
        : name{name}, data{data}, n_elems{1}, elem_size{sizeof(T)},
          type{jit_types<T>::value}, array{false} {}

    template<typename T>
    jit_param(std::string_view name, T *data, std::size_t len)
        : name{name}, data{data}, n_elems{len}, elem_size{sizeof(T)},
          type{jit_types<T>::value}, array{true} {}

    uint64_t hash(uint64_t seed = hash_seed_default) {
        return op::f2c::hash(data, n_elems * elem_size, seed);
    }

    std::string format_type() {
        switch (type) {
            case param_type::i32:     return "int";
            case param_type::i64:     return "int64_t";
            case param_type::f32:     return "float";
            case param_type::f64:     return "double";
            case param_type::logical: return "bool";
        }

        __builtin_unreachable();
    }

    std::string format_value() {
        std::ostringstream os;
        if (array) os << "{ ";

        for (std::size_t i = 0; i < n_elems; ++i) {
            char *elem = (char *)data + elem_size * i;

            switch (type) {
                case param_type::i32:     os << *((int *)elem); break;
                case param_type::i64:     os << *((int64_t *)elem); break;
                case param_type::f32:     os << std::hexfloat << *((float *)elem); break;
                case param_type::f64:     os << std::hexfloat << *((double *)elem); break;
                case param_type::logical: os << std::boolalpha << *((bool *)elem); break;
            }

            if (array && i < n_elems - 1) os << ", ";
        }

        if (array) os << " }";
        return os.str();
    }

    std::string format() {
        std::ostringstream os;

        os << "static constexpr " << format_type() << " " << name;
        if (array) { os << "[" << n_elems << "]"; }
        os << " = " << format_value() << ";" << std::endl;

        return os.str();
    }
};

struct hash_info {
    std::size_t count = 0;
    bool jit_started = false;
};

class kernel_info {
private:
    std::string name;
    const void* kernel;
    cudaFuncAttributes kernel_attrs;

    std::vector<jit_param> params;
    std::string src;

    std::mutex jit_kernels_mutex;
    std::unordered_map<uint64_t, jit_kernel> jit_kernels;

    std::unordered_map<uint64_t, hash_info> hash_infos;

    bool profile;

    bool is_jit_candidate() {
        return kernel_attrs.numRegs > 32;
    }

    uint64_t hash_params(uint64_t seed = hash_seed_default) {
        uint64_t hash = seed;

        for (auto& param : params)
            hash = param.hash(hash);

        return hash;
    }

    std::string format_params() {
        auto src = std::string();

        for (auto& param : params)
            src += param.format();

        return src;
    }

    void compile(uint64_t hash) {
        std::string jit_src = std::string("#include <op_f2c_prelude.h>\n") +
                              std::string("#include <op_f2c_params.h>\n") +
                              std::string("\nnamespace f2c = op::f2c;\n") +
                              format_params() + src;

        auto do_compile = [&](auto jit_src, auto hash) {
            const char *headers[] = { op_f2c_prelude_data, op_f2c_params_data };
            const char *header_names[] = { "op_f2c_prelude.h", "op_f2c_params.h" };

            nvrtcProgram prog;
            NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, jit_src.c_str(), name.c_str(),
                                               2, headers, header_names));

            const char *opts[] = {
                // "-use_fast_math",
                // "--generate-line-info",
                "--std=c++20",
                "-arch=sm_90",
                "-minimal",
                "-default-device"
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

            std::scoped_lock lock(jit_kernels_mutex);
            auto [it, inserted] = jit_kernels.emplace(std::piecewise_construct,
                    std::forward_as_tuple(hash), std::forward_as_tuple(cubin, name));

            assert(inserted);
        };

        if (jit_seq_compile) {
            do_compile(jit_src, hash);
            return;
        }

        std::thread compilation_thread(do_compile, jit_src, hash);
        compilation_thread.detach();
    }

public:
    kernel_info(const kernel_info&) = delete;
    kernel_info(std::string_view name, const void *kernel, std::string_view src)
        : name{name}, kernel{kernel}, src{src} {
        jit_init();
        // profile = name == "op2_k_flux_2_vflux_edge_main_wrapper";
        profile = name == "op2_k_flux_18_grad_edge_work_1_main_wrapper";

        CUDA_SAFE_CALL(cudaFuncGetAttributes(&kernel_attrs, kernel));
    }

    template<typename T>
    void add_param(std::string_view name, T *data) {
        params.emplace_back(name, data);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, std::size_t len) {
        params.emplace_back(name, data, len);
    }

    bool has_compiled(uint64_t *hash) {
        if (!jit_enable || !is_jit_candidate()) {
            *hash = 0;
            return false;
        }

        auto params_hash = hash_params();
        *hash = params_hash;

        std::scoped_lock lock(jit_kernels_mutex);
        auto kernel_elem = jit_kernels.find(params_hash);

        return kernel_elem != jit_kernels.end();
    }

    void invoke(uint64_t params_hash, int num_blocks, int block_size, void **args, void **args_jit) {
        if (!jit_enable || !is_jit_candidate()) {
            op_timing2_next("Offline Kernel");
            if (profile) CUDA_SAFE_CALL(cudaProfilerStart());
            CUDA_SAFE_CALLN(cudaLaunchKernel(kernel, num_blocks, block_size, args, 0, 0));
            if (profile) CUDA_SAFE_CALL(cudaProfilerStop());

            CUDA_SAFE_CALLN(cudaPeekAtLastError());
            // CUDA_SAFE_CALLN(cudaStreamSynchronize(0));
            return;
        }

        auto [hash_elem, inserted] = hash_infos.insert({params_hash, hash_info()});
        hash_elem->second.count++;

        op_timing2_next("JIT Lookup");
        jit_kernels_mutex.lock();
        auto kernel_elem = jit_kernels.find(params_hash);

        if (kernel_elem != jit_kernels.end()) {
            // std::printf("using jit %s (hash %lx)\n", name.c_str(), params_hash);
            auto& jk = kernel_elem->second;
            jit_kernels_mutex.unlock();

            op_timing2_next("JIT Kernel");
            jk.invoke(num_blocks, block_size, args_jit);
            return;
        }

        jit_kernels_mutex.unlock();

        op_timing2_next("JIT Compilation");

        // Launch async compilation
        if (hash_elem->second.count > 8 && !hash_elem->second.jit_started) {
            std::printf("compiling %s for hash %lx\n", name.c_str(), params_hash);
            hash_elem->second.jit_started = true;
            compile(params_hash);
        }

        // std::printf("using offline %s for hash %lx (invocation %ld)\n",
        //         name.c_str(), params_hash, hash_elem->second);

        op_timing2_next("Offline Kernel");
        if (profile) CUDA_SAFE_CALL(cudaProfilerStart());
        CUDA_SAFE_CALLN(cudaLaunchKernel(kernel, num_blocks, block_size, args, 0, 0));
        if (profile) CUDA_SAFE_CALL(cudaProfilerStop());

        CUDA_SAFE_CALLN(cudaPeekAtLastError());
        // CUDA_SAFE_CALLN(cudaStreamSynchronize(0));
    }
};

} // namespace op::f2c
