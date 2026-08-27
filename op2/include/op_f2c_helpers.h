#pragma once

#include <extern/rapidhash.h>
#include <op_lib_cpp.h>
#include <op_profile.h>
#include <op_gpu_shims.h>
#include <op_hier_smem_cache.h>
#include <op_hier_smem_plan.h>

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

class ExecutionSchedule {
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

    ExecutionSchedule(Kind kind, op_set set, op_plan *plan,
                      bool separate_owned)
        : m_kind{kind}, m_set{set}, m_plan{plan},
          m_separate_owned{separate_owned} {}

public:
    static ExecutionSchedule direct(op_set set) {
        return ExecutionSchedule{Kind::direct, set, nullptr, false};
    }

    static ExecutionSchedule atomics(op_set set, bool separate_owned) {
        return ExecutionSchedule{Kind::atomics, set, nullptr, separate_owned};
    }

    static ExecutionSchedule color2(op_set set, op_plan *plan) {
        return ExecutionSchedule{Kind::color2, set, plan, false};
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

    bool wait_before(int index) const {
        assert(index >= 0 && index < size());

        switch (m_kind) {
        case Kind::direct:
            return false;
        case Kind::atomics:
            return index == 1;
        case Kind::color2:
            return index == m_plan->ncolors_core;
        }

        assert(false);
        return false;
    }

    bool process_globals_after(int index) const {
        assert(index >= 0 && index < size());

        switch (m_kind) {
        case Kind::direct:
            return index == 0;
        case Kind::atomics:
            return index == 1;
        case Kind::color2:
            return index == m_plan->ncolors_owned - 1;
        }

        assert(false);
        return false;
    }

    int set_stride() const {
        int size = m_set->size;
        if (m_kind != Kind::direct)
            size += m_set->exec_size;

        return (size + 31) & ~31;
    }

    int *color_reorder() const {
        return m_kind == Kind::color2 ? m_plan->col_reord : nullptr;
    }
};

enum class KernelVariant {
    baseline,
    staged,
};

struct KernelExecutionOptions {
    KernelVariant variant = KernelVariant::baseline;
    int shared_bytes = 0;
    bool plan_hier_smem_for_testing = false;

    static KernelExecutionOptions hierarchical_test() {
        return {KernelVariant::staged, 0, true};
    }
};

struct KernelExecution {
    KernelVariant variant;
    JitKernel *jit_kernel;
    ExecutionSchedule schedule;
    int block_size;
    int block_limit;
    int max_blocks;
    int shared_bytes;
    const HierSmemPlan *hier_smem_plan;
    HierSmemPlanDeviceView hier_smem_device;
    HierSmemFallbackReason hier_smem_reason;

    int num_blocks(int section_index) const {
        if (hier_smem_plan != nullptr) {
            int blocks = hier_smem_plan->section_chunk_offsets[section_index + 1] -
                         hier_smem_plan->section_chunk_offsets[section_index];
            return std::min(blocks, block_limit);
        }

        auto section = schedule[section_index];
        int blocks = (section.size() + block_size - 1) / block_size;
        return std::min(blocks, block_limit);
    }

    int dynamic_shared_bytes(int section_index) const {
        if (hier_smem_plan == nullptr)
            return shared_bytes;

        auto bytes = hier_smem_plan->section_shared_bytes[section_index];
        assert(bytes <= static_cast<std::size_t>(INT32_MAX));
        return static_cast<int>(bytes);
    }
};

class ExecutionPolicy {
public:
    enum class Kind {
        direct,
        atomics,
        color2,
    };

private:
    Kind m_kind;
    bool m_gbl_inc_atomic = false;
    int m_part_size = -1;
    std::vector<int> m_indirect_dats;

    ExecutionPolicy(Kind kind, bool gbl_inc_atomic, int part_size,
                    std::vector<int> indirect_dats)
        : m_kind{kind}, m_gbl_inc_atomic{gbl_inc_atomic},
          m_part_size{part_size}, m_indirect_dats{std::move(indirect_dats)} {}

public:
    static ExecutionPolicy direct(bool gbl_inc_atomic) {
        return {Kind::direct, gbl_inc_atomic, -1, {}};
    }

    static ExecutionPolicy atomics(bool gbl_inc_atomic) {
        return {Kind::atomics, gbl_inc_atomic, -1, {}};
    }

    template<std::size_t N>
    static ExecutionPolicy color2(const std::array<int, N>& indirect_dats,
                                  int part_size, bool gbl_inc_atomic) {
        return {Kind::color2, gbl_inc_atomic, part_size,
                {indirect_dats.begin(), indirect_dats.end()}};
    }

    Kind kind() const { return m_kind; }
    bool gbl_inc_atomic() const { return m_gbl_inc_atomic; }
    int part_size() const { return m_part_size; }
    std::size_t nargs() const { return m_indirect_dats.size(); }
    int ninds() const {
        int max_ind = -1;
        for (int ind : m_indirect_dats)
            max_ind = std::max(max_ind, ind);
        return max_ind + 1;
    }
    int *indirect_dats() { return m_indirect_dats.data(); }
};

struct LaunchContext {
    int global_stride;
    unsigned opt_flags;
    int start;
    int end;
    int set_stride;
    int *color_reorder;

    struct {
        HierSmemPlanDeviceView plan;
        int chunk_begin;
        int chunk_end;
    } staged;
};

struct GlobalInitContext {
    int block_size;
    int max_blocks;
    int global_stride;
};

template<std::size_t OfflineN, std::size_t JitN>
struct KernelArguments {
    std::array<void *, OfflineN> offline;
    std::array<void *, JitN> jit;
};

template<std::size_t OfflineN, std::size_t JitN>
KernelArguments(std::array<void *, OfflineN>, std::array<void *, JitN>)
    -> KernelArguments<OfflineN, JitN>;

struct KernelInvocationResult {
    bool used_jit;
    KernelVariant variant;
    int block_size;
    int max_blocks;
    HierSmemFallbackReason hier_smem_reason = HierSmemFallbackReason::none;
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
    ExecutionPolicy m_policy;
    KernelImplementation m_baseline;
    std::unique_ptr<KernelImplementation> m_staged;
    std::optional<HierSmemStagingDescriptor> m_staging_descriptor;
    detail::HierSmemPlanCache m_hier_smem_cache;
    std::optional<std::size_t> m_hier_smem_capacity;
    bool m_plan_owner_registered = false;
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

    static void release_hier_smem_plans_callback(void *owner) {
        static_cast<KernelInfo *>(owner)->m_hier_smem_cache.clear();
    }

    void register_plan_owner() {
        if (m_plan_owner_registered)
            return;

        register_hier_smem_plan_owner(
            this, &KernelInfo::release_hier_smem_plans_callback);
        m_plan_owner_registered = true;
    }

public:
    KernelInfo(const KernelInfo&) = delete;
    KernelInfo(std::string_view profile_name, std::string_view profile_target,
               std::string_view profile_variant, ExecutionPolicy policy,
               std::string_view name, const void *kernel, std::string_view src)
        : m_name{name}, m_profile_name{profile_name},
          m_profile_target{profile_target}, m_profile_variant{profile_variant},
          m_policy{std::move(policy)}, m_baseline{name, kernel, src} {
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

        if (m_plan_owner_registered)
            unregister_hier_smem_plan_owner(this);
        m_hier_smem_cache.clear();
    }

    void register_staged_variant(std::string_view name, const void *kernel,
                                 std::string_view src,
                                 std::optional<HierSmemStagingDescriptor>
                                     descriptor = std::nullopt) {
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
        m_staging_descriptor = descriptor;
    }

    HierSmemPlanCacheStatistics hier_smem_plan_cache_statistics() const {
        return m_hier_smem_cache.statistics();
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

    // Return the staged wrapper's usable dynamic shared-memory capacity.
    // OP2 fixes the device at initialization, so this is resolved once.
    std::size_t hier_smem_device_capacity() {
        assert(m_staged != nullptr);
        if (m_hier_smem_capacity.has_value())
            return *m_hier_smem_capacity;

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
            m_staged->offline_attrs.sharedSizeBytes);
        m_hier_smem_capacity = total > static_bytes ? total - static_bytes : 0;
        return *m_hier_smem_capacity;
    }

    // Reject staged wrappers that omit an active indirect increment.
    static bool all_indirect_increments_covered(
        std::span<const op_arg> args,
        const HierSmemStagingDescriptor& descriptor) {
        std::vector<bool> covered(args.size(), false);
        for (const auto& arg : descriptor.args) {
            assert(arg.arg_index >= 0 &&
                   static_cast<std::size_t>(arg.arg_index) < args.size());
            covered[static_cast<std::size_t>(arg.arg_index)] = true;
        }

        for (std::size_t i = 0; i < args.size(); ++i) {
            const op_arg& arg = args[i];
            if (arg.opt != 0 && arg.argtype == OP_ARG_DAT &&
                arg.acc == OP_INC && arg.idx >= 0 && !covered[i])
                return false;
        }

        return true;
    }

    // Look up or build the plan for the current loop configuration.
    const detail::HierSmemPlanCacheEntry& get_hier_smem_plan(
        op_set set, std::span<const op_arg> args,
        std::span<const ExecutionSection> sections, int block_size) {
        assert(m_staging_descriptor.has_value());
        register_plan_owner();

        HierSmemPlanOptions plan_options{
            block_size, OP_part_size, hier_smem_device_capacity()};
        auto key = detail::make_hier_smem_plan_key(
            set, args, static_cast<int>(sections.size()),
            *m_staging_descriptor, plan_options);

        return m_hier_smem_cache.get_or_build(std::move(key), [&]() {
            if (!all_indirect_increments_covered(args, *m_staging_descriptor))
                return HierSmemPlanBuildResult{
                    HierSmemFallbackReason::incompatible_argument,
                    std::nullopt};

            return build_hier_smem_plan(
                set, args, sections, *m_staging_descriptor, plan_options);
        });
    }

    KernelExecution prepare(op_set set, op_arg *args, int nargs,
                            ExecutionSchedule schedule,
                            KernelExecutionOptions options) {
        // Resolve the common physical launch policy first.
        int max_section_size = 0;
        for (int i = 0; i < schedule.size(); ++i)
            max_section_size = std::max(max_section_size, schedule[i].size());

        auto [block_limit, block_size] = get_launch_config(nullptr, max_section_size);
        block_limit = std::min(
            block_limit,
            ::getBlockLimitWithPolicy(args, nargs, block_size, m_name.c_str(),
                                      m_policy.gbl_inc_atomic()));

        KernelVariant variant = options.variant;
        const HierSmemPlan *hier_smem_plan = nullptr;
        HierSmemPlanDeviceView hier_smem_device;
        HierSmemFallbackReason hier_smem_reason =
            HierSmemFallbackReason::none;

        // Step 5 keeps planning behind an explicit test-only activation.
        if (options.plan_hier_smem_for_testing) {
            if (variant != KernelVariant::staged || m_staged == nullptr ||
                !m_staging_descriptor.has_value() ||
                m_policy.kind() != ExecutionPolicy::Kind::atomics) {
                variant = KernelVariant::baseline;
                hier_smem_reason =
                    HierSmemFallbackReason::incompatible_argument;
            } else {
                std::array<ExecutionSection, 3> sections;
                assert(schedule.size() <= static_cast<int>(sections.size()));
                for (int i = 0; i < schedule.size(); ++i)
                    sections[static_cast<std::size_t>(i)] = schedule[i];

                const auto& cached = get_hier_smem_plan(
                    set, std::span<const op_arg>{args,
                                                static_cast<std::size_t>(nargs)},
                    std::span<const ExecutionSection>{
                        sections.data(),
                        static_cast<std::size_t>(schedule.size())},
                    block_size);
                hier_smem_reason = cached.reason();
                if (cached) {
                    hier_smem_plan = cached.plan();
                    hier_smem_device = cached.device_view();
                } else {
                    variant = KernelVariant::baseline;
                }
            }
        }

        // Reduction scratch uses the selected plan's capped physical grid.
        int max_blocks = 0;
        for (int i = 0; i < schedule.size(); ++i) {
            int section_blocks = 0;
            if (hier_smem_plan == nullptr) {
                section_blocks =
                    (schedule[i].size() + block_size - 1) / block_size;
            } else {
                section_blocks =
                    hier_smem_plan->section_chunk_offsets[i + 1] -
                    hier_smem_plan->section_chunk_offsets[i];
            }
            max_blocks = std::max(max_blocks, section_blocks);
        }

        max_blocks = std::min(max_blocks, block_limit);

        KernelExecution execution{
            variant, nullptr, schedule, block_size, block_limit, max_blocks,
            options.shared_bytes, hier_smem_plan, hier_smem_device,
            hier_smem_reason};

        for (auto& param : m_params)
            param.update(args, nargs, execution);

        execution.jit_kernel = get_kernel(implementation(variant));

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
        auto& impl = implementation(execution.variant);

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

    template<KernelVariant Variant, typename Bindings>
    void bind_and_launch(const KernelExecution& execution, int section_index,
                         int num_blocks,
                         LaunchContext& launch, op_arg *args,
                         Bindings& bindings) {
        auto kernel_args =
            bindings.template make_arguments<Variant>(launch, args);
        launch_section(execution, section_index, num_blocks,
                       kernel_args.offline.data(),
                       kernel_args.jit.data());
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
        int n_exec = op_mpi_halo_exchanges_grouped(set, nargs, args, 2);

        if (n_exec == 0) {
            op_profile_exit();
            op_profile_exit();

            op_mpi_wait_all_grouped(nargs, args, 2);
            reduce_mpi_globals(args, nargs);
            op_mpi_set_dirtybit_cuda(nargs, args);
            op_profile_exit();

            return {false, options.variant, 0, 0};
        }

        bool global_reduction = has_global_reduction(args, nargs);
        bool global_output = has_global_output(args, nargs);
        ExecutionSchedule schedule = ExecutionSchedule::direct(set);

        switch (m_policy.kind()) {
        case ExecutionPolicy::Kind::direct:
            schedule = ExecutionSchedule::direct(set);
            break;
        case ExecutionPolicy::Kind::atomics:
            schedule = ExecutionSchedule::atomics(set, global_reduction);
            break;
        case ExecutionPolicy::Kind::color2: {
            op_profile_enter("Plan");

            int part_size = m_policy.part_size() >= 0
                                ? m_policy.part_size()
                                : OP_part_size;
            if (m_policy.nargs() != static_cast<std::size_t>(nargs) ||
                m_policy.ninds() == 0) {
                std::fprintf(stderr,
                             "error: invalid color2 indirect dat mapping (in %s)\n",
                             m_name.c_str());
                std::exit(1);
            }

            op_plan *plan = op_plan_get_stage(
                m_profile_name.c_str(), set, part_size, nargs, args,
                m_policy.ninds(), m_policy.indirect_dats(), OP_COLOR2);
            schedule = ExecutionSchedule::color2(set, plan);

            op_profile_exit();
            break;
        }
        }

        op_profile_next("Get Kernel");
        auto execution = prepare(set, args, nargs, schedule, options);
        op_profile_exit();

        op_profile_enter("Prepare GBLs");
        int global_stride = execution.block_size * execution.max_blocks;
        prepareDeviceGblsWithPolicy(args, nargs, global_stride,
                                    m_policy.gbl_inc_atomic());

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
                op_mpi_wait_all_grouped(nargs, args, 2);
                op_profile_next("Kernel");
            }

            auto section = schedule[section_index];
            if (section.size() > 0) {
                LaunchContext launch{
                    global_stride,
                    0,
                    section.start,
                    section.end,
                    schedule.set_stride(),
                    schedule.color_reorder(),
                    {}};

                if (execution.hier_smem_plan != nullptr) {
                    launch.staged.plan = execution.hier_smem_device;
                    launch.staged.chunk_begin =
                        execution.hier_smem_plan
                            ->section_chunk_offsets[section_index];
                    launch.staged.chunk_end =
                        execution.hier_smem_plan
                            ->section_chunk_offsets[section_index + 1];
                }

                int num_blocks = execution.num_blocks(section_index);
                if (execution.variant == KernelVariant::baseline)
                    bind_and_launch<KernelVariant::baseline>(
                        execution, section_index, num_blocks, launch, args,
                        bindings);
                else
                    bind_and_launch<KernelVariant::staged>(
                        execution, section_index, num_blocks, launch, args,
                        bindings);
            }

            if (global_output &&
                schedule.process_globals_after(section_index)) {
                op_profile_next("Process GBLs");
                exit_sync |= processDeviceGblsWithPolicy(
                    args, nargs, global_stride, global_stride,
                    m_policy.gbl_inc_atomic());
                op_profile_next("Kernel");
            }
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

        return {execution.jit_kernel != nullptr, execution.variant,
                execution.block_size, execution.max_blocks,
                execution.hier_smem_reason};
    }
};

} // namespace op::f2c
