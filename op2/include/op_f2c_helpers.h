#pragma once

#include <extern/rapidhash.h>
#include <op_timing2.h>

#include <nvrtc.h>
#include <cuda.h>

#include <array>
#include <vector>
#include <unordered_map>
#include <string>
#include <cassert>
#include <sstream>


#define NVRTC_SAFE_CALL(x)                                                     \
    do {                                                                       \
        nvrtcResult result = x;                                                \
        if (result != NVRTC_SUCCESS) {                                         \
            const char *msg = nvrtcGetErrorString(result);                     \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,    \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

#define CU_SAFE_CALL(x)                                                        \
    do {                                                                       \
        CUresult result = x;                                                   \
        if (result != CUDA_SUCCESS) {                                          \
            const char *msg;                                                   \
            cuGetErrorName(result, &msg);                                      \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,    \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

#define CUDA_SAFE_CALL(x)                                                      \
    do {                                                                       \
        cudaError_t result = x;                                                \
        if (result != cudaSuccess) {                                           \
            const char *msg = cudaGetErrorString(result);                      \
            fprintf(stderr, "error: " #x " failed with %s at %s:%d\n", msg,    \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while(0)


namespace op::f2c {

constexpr uint64_t hash_seed_default = RAPID_SEED;

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
    CUmodule module;
    CUfunction kernel;

    jit_kernel(const jit_kernel&) = delete;

    jit_kernel(jit_kernel&& other) : module{other.module}, kernel{other.kernel} {
        other.module = nullptr;
        other.kernel = nullptr;
    };

    jit_kernel(const char *cubin, const std::string &name) {
        CU_SAFE_CALL(cuModuleLoadData(&module, cubin));
        CU_SAFE_CALL(cuModuleGetFunction(&kernel, module, name.c_str()));
    }

    /*
    ~jit_kernel() {
        CU_SAFE_CALL(cuModuleUnload(module));
    }
    */

    void invoke(int num_blocks, int block_size, void **args) {
        CU_SAFE_CALL(cuLaunchKernel(kernel, num_blocks, 1, 1, block_size, 1, 1, 0,
                                      NULL, args, 0));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
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

class kernel_info {
private:
    std::string name;
    const void* kernel;

    std::vector<jit_param> params;
    std::string src;

    std::unordered_map<uint64_t, jit_kernel> jit_kernels;
    std::unordered_map<uint64_t, std::size_t> hash_counts;

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

    jit_kernel& compile(uint64_t hash) {
        std::string jit_src = std::string("#include <op_f2c_prelude.h>\n") +
                              std::string("#include <op_f2c_params.h>\n") +
                              std::string("\nnamespace f2c = op::f2c;\n") +
                              format_params() + src;

        const char *headers[] = { op_f2c_prelude_data, op_f2c_params_data };
        const char *header_names[] = { "op_f2c_prelude.h", "op_f2c_params.h" };

        nvrtcProgram prog;
        NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, jit_src.c_str(), name.c_str(),
                                           2, headers, header_names));

        const char *opts[] = {
            "-use_fast_math",
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

        auto [it, inserted] = jit_kernels.emplace(std::piecewise_construct,
                std::forward_as_tuple(hash), std::forward_as_tuple(cubin, name));

        assert(inserted);

        NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
        delete[] cubin;

        return it->second;
    }

public:
    kernel_info(const kernel_info&) = delete;
    kernel_info(std::string_view name, const void *kernel, std::string_view src)
        : name{name}, kernel{kernel}, src{src} {}

    template<typename T>
    void add_param(std::string_view name, T *data) {
        params.emplace_back(name, data);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, std::size_t len) {
        params.emplace_back(name, data, len);
    }

    void invoke(int num_blocks, int block_size, void **args) {
        op_timing2_next("Hash params");
        auto params_hash = hash_params();

        auto [hash_elem, inserted] = hash_counts.insert({params_hash, 1});
        if (!inserted) hash_elem->second++;

        auto kernel_elem = jit_kernels.find(params_hash);
        if (kernel_elem != jit_kernels.end()) {
            // std::printf("using jit %s (hash %lx)\n", name.c_str(), params_hash);
            //
            op_timing2_next("Kernel");
            kernel_elem->second.invoke(num_blocks, block_size, args);
            return;
        }

        if (hash_elem->second <= 3) {
            // std::printf("using offline %s for hash %lx (invocation %ld)\n",
            //         name.c_str(), params_hash, hash_elem->second);
            op_timing2_next("Kernel");
            CUDA_SAFE_CALL(cudaLaunchKernel(kernel, num_blocks, block_size, args, 0, 0));
            return;
        }

        op_timing2_next("JIT Compilation");
        //std::printf("compiling %s for hash %lx\n", name.c_str(), params_hash);
        auto& kernel = compile(params_hash);

        op_timing2_next("Kernel");
        kernel.invoke(num_blocks, block_size, args);
    }
};

} // namespace op::f2c
