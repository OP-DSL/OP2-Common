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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
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

    int registers() {
        ensure_loaded();
        int count = 0;
        CU_SAFE_CALL(gpuDrvFuncGetAttribute(&count, gpuDrvFuncAttributeNumRegs, m_kernel));
        return count;
    }

    // Blocks of block_size threads a multiprocessor holds, by registers.
    int occupancy(int block_size) { return config().occupancy(registers(), block_size); }

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

// One parameter of a grid-stride JIT wrapper, in binding order: a dat's or
// map's device pointer, a global's, a per-call value, or the launch range;
// then op2_body's scalar READ globals, which the JIT wrapper bakes in, with
// their arguments.
struct FusionParam {
    enum Kind { buffer, global, value, start, end, scalar };

    const char *type;
    const char *name;
    Kind kind;
    int arg = -1;
};

// What a fused kernel needs of a strategy: its element function, named
// within the strategy's declarations, and its wrapper's parameters, of which
// the function takes all but the range.
struct FusionSource {
    const char *body;
    std::span<const FusionParam> params;
};

// What a strategy's kernel is built from: the offline function, if any, and
// the source a specialised one is compiled from, its declarations ahead of
// its wrapper; fusion when a fused kernel can take it.
struct KernelSource {
    std::string_view name;
    const void *offline = nullptr;
    std::string_view source;
    std::string_view decls = {};
    const FusionSource *fusion = nullptr;
};

// --- Parameters ---------------------------------------------------------------------

enum class ParamType {
    i32,
    i64,
    f32,
    f64,
    logical,
};

// Where a parameter's value comes from: the application, the call, or the
// strategy.
enum class ParamSource {
    external,          // a constant the application sets
    scalar_arg,        // a scalar global argument
    dat_stride,        // a dat's SoA stride
    dat_dim,           // a dat's run-time dimension
    global_stride,     // the launch's per-thread global stride
    block_size,        // the launch's block size
    unstaged_singles,  // the strategy's unstaged singles switch
    wavefront,         // the strategy's wavefront switch
};

// What a strategy fixes for its kernel: the launch, and its variant's
// switches.  Passed to each acquire and upload, never kept between them.
struct LaunchValues {
    int global_stride = 0;
    int block_size = 0;
    bool unstaged_singles = false;
    bool wavefront = false;
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
    // Where an application constant is read from at update.
    void *m_data;
    // The application's or the call's value as of the last update, or a
    // queued call's, restored; a launch value is held in m_launch, applied
    // before each use.
    std::vector<std::byte> m_value;
    int m_launch = 0;
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
          m_hash_device_ptr{hash_device_ptr} {
        if (!launch())
            m_value.resize(len * sizeof(T));
    }

    bool scalar() const { return m_source == ParamSource::scalar_arg; }

    // Whether the strategy gives the value at each use.
    bool launch() const {
        return m_source == ParamSource::global_stride ||
               m_source == ParamSource::block_size ||
               m_source == ParamSource::unstaged_singles ||
               m_source == ParamSource::wavefront;
    }

    // The application's and the call's values.
    void update(op_arg *args, int nargs) {
        switch (m_source) {
        case ParamSource::external:
            std::memcpy(m_value.data(), m_data, m_value.size());
            return;
        case ParamSource::scalar_arg:
            assert(m_arg_index >= 0 && m_arg_index < nargs);
            assert(args[m_arg_index].data != nullptr);
            std::memcpy(m_value.data(), args[m_arg_index].data, m_elem_size);
            return;
        case ParamSource::dat_stride: {
            assert(m_arg_index >= 0 && m_arg_index < nargs);
            assert(m_elem_size == sizeof(int) && m_n_elems == 1);
            int size = getSetSizeFromOpArg(&args[m_arg_index]);
            size = (size + 31) & ~31;
            std::memcpy(m_value.data(), &size, sizeof(int));
            return;
        }
        case ParamSource::dat_dim: {
            assert(m_arg_index >= 0 && m_arg_index < nargs);
            assert(m_elem_size == sizeof(int) && m_n_elems == 1);
            int dim = args[m_arg_index].dim;
            std::memcpy(m_value.data(), &dim, sizeof(int));
            return;
        }
        default:
            (void)nargs;
            return;
        }
    }

    // The value bytes update gave, kept with a queued call and given back
    // before it runs.
    void save(std::vector<std::byte>& out) const {
        out.insert(out.end(), m_value.begin(), m_value.end());
    }

    const std::byte *restore(const std::byte *in) {
        std::memcpy(m_value.data(), in, m_value.size());
        return in + m_value.size();
    }

    // The strategy's values, for a launch-sourced parameter.
    void apply(const LaunchValues& values) {
        switch (m_source) {
        case ParamSource::global_stride:    m_launch = values.global_stride; return;
        case ParamSource::block_size:       m_launch = values.block_size; return;
        case ParamSource::unstaged_singles: m_launch = values.unstaged_singles; return;
        case ParamSource::wavefront:        m_launch = values.wavefront; return;
        default:                            return;
        }
    }

    const void *data() const {
        return m_value.empty() ? static_cast<const void *>(&m_launch)
                               : static_cast<const void *>(m_value.data());
    }

    uint64_t hash() {
        m_hash_last = hash_bytes(data(), m_n_elems * m_elem_size);
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
        CUDA_SAFE_CALL(gpuMemcpyAsync(m_data_d, data(), m_elem_size * m_n_elems,
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
            const char *elem = static_cast<const char *>(data()) + m_elem_size * i;
            switch (m_type) {
            case ParamType::i32:     os << *((const int *)elem); break;
            case ParamType::i64:     os << *((const int64_t *)elem); break;
            case ParamType::f32:     os << std::hexfloat << *((const float *)elem); break;
            case ParamType::f64:     os << std::hexfloat << *((const double *)elem); break;
            case ParamType::logical: os << std::boolalpha << *((const bool *)elem); break;
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
        // When it was last asked for, on m_clock.
        std::uint64_t used = 0;
        bool started = false;
        std::thread thread;
    };

    // Past this many entries the least recently asked-for half of those not
    // started go, so values that change every call stay bounded.
    static constexpr std::size_t max_compiles = 256;

    gpuFuncAttributes_t m_offline_attrs{};
    int m_offline_max_dynamic_shared_bytes = 0;
    std::unordered_map<uint64_t, JitKernel> m_kernels;
    std::unordered_map<uint64_t, Compile> m_compiles;
    std::uint64_t m_clock = 0;
    std::mutex m_mutex;

    // The entry for a hash, marked as used now.
    Compile& compile_entry(uint64_t hash) {
        auto [found, inserted] = m_compiles.try_emplace(hash);
        found->second.used = ++m_clock;
        if (inserted && m_compiles.size() > max_compiles)
            evict();
        return found->second;
    }

    // Drop the least recently used half of the entries not started; the
    // newest is never among them.
    void evict() {
        std::vector<std::pair<std::uint64_t, uint64_t>> idle;
        for (const auto& [hash, compile] : m_compiles)
            if (!compile.started)
                idle.emplace_back(compile.used, hash);
        auto middle = idle.begin() + static_cast<std::ptrdiff_t>(idle.size() / 2);
        std::nth_element(idle.begin(), middle, idle.end());
        for (auto entry = idle.begin(); entry != middle; ++entry)
            m_compiles.erase(entry->second);
    }

public:
    JitKernels() = default;
    JitKernels(const JitKernels&) = delete;
    ~JitKernels() { join(); }

    void register_offline(const void *kernel) {
        if (kernel != nullptr)
            CUDA_SAFE_CALL(gpuFuncGetAttributes(&m_offline_attrs, kernel));
    }

    // The offline kernel's registers; 0 without one.
    int offline_registers() const { return m_offline_attrs.numRegs; }

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

    // The parameters' hash under a strategy's launch values, which stay
    // applied for a compile that follows; without the scalar arguments for
    // a kernel that takes them as arguments.
    uint64_t hash_params(const LaunchValues& launch, bool scalars = true) {
        uint64_t hash = hash_seed_default;
        for (auto& param : m_params) {
            param.apply(launch);
            if (scalars || !param.scalar())
                hash = op::gpu::hash(param.hash(), hash);
        }
        return hash;
    }

    std::string format_params(bool scalars = true) const {
        std::string src;
        for (const auto& param : m_params)
            if (scalars || !param.scalar())
                src += param.format();
        return src;
    }

    // Compile source into kernels' entry for hash, on a thread of its own;
    // name is the kernel's, loop the compile's for messages.
    static std::thread compile(JitKernels& kernels, const JitHeaders& headers,
                               std::string name, std::string source, uint64_t hash,
                               std::string loop) {
        ++active_threads;
        auto do_compile = [&kernels, &headers, name = std::move(name),
                           loop = std::move(loop)](std::string jit_src, uint64_t hash) {
            gpuRtcProgram_t prog;
            NVRTC_SAFE_CALL(gpuRtcCreateProgram(
                &prog, jit_src.c_str(), name.c_str(),
                static_cast<int>(headers.data.size()), headers.data.data(),
                headers.names.data()));

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
                             name.c_str(), loop.c_str());
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

        return std::thread(do_compile, std::move(source), hash);
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

    // What every compile's source starts with.
    static std::string preamble(const JitHeaders& headers) {
        return std::string("#include <kernel/prelude.h>\n"
                           "#include <kernel/strategy.h>\n") +
               (headers.params ? "#include <op_f2c_params.h>\n" : "") +
               "\nnamespace f2c = op::f2c;\n";
    }

    const JitHeaders& headers() const { return m_headers; }

    // The parameters' hash and constants under a strategy's launch values,
    // for a fused kernel, which takes the scalar arguments as arguments.
    uint64_t fusion_hash(const LaunchValues& launch) { return hash_params(launch, false); }
    std::string fusion_source(const LaunchValues& launch) {
        hash_params(launch, false);
        return format_params(false);
    }

    // The kernel compiled for hash, else its compile started from
    // make_source() once asked OP_JIT_MIN_CALLS times (at once for
    // Urgency::now) and joined when wait is set or under
    // OP_JIT_SEQ_COMPILE; null while it is not ready.
    template<typename MakeSource>
    static Acquired acquire_compiled(JitKernels& kernels, const JitHeaders& headers,
                                     std::string_view name, uint64_t hash,
                                     op::loop::Urgency urgency, bool wait,
                                     MakeSource&& make_source, std::string_view loop) {
        const auto& run = config();
        {
            std::scoped_lock lock(kernels.m_mutex);
            auto found = kernels.m_kernels.find(hash);
            if (found != kernels.m_kernels.end())
                return {op::loop::Status::ready, &found->second};
        }

        auto& compile = kernels.compile_entry(hash);
        ++compile.count;
        bool due = urgency != op::loop::Urgency::later ||
                   compile.count >= static_cast<std::size_t>(run.min_calls);
        if (due && !compile.started && (wait || active_threads < run.max_threads)) {
            if (run.debug)
                std::printf("compiling %s for hash %lx\n", std::string{name}.c_str(), hash);
            compile.started = true;
            compile.thread = JitService::compile(kernels, headers, std::string{name},
                                                 make_source(), hash, std::string{loop});
        }
        if (compile.started && (wait || run.seq_compile)) {
            if (compile.thread.joinable())
                compile.thread.join();
            std::scoped_lock lock(kernels.m_mutex);
            return {op::loop::Status::ready, &kernels.m_kernels.at(hash)};
        }
        return {op::loop::Status::pending, nullptr};
    }

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

    // A value each strategy decides (global stride, block size, singles,
    // wavefront), given to every acquire and upload; symbol is the copy an
    // offline kernel reads, if any.
    void add_launch_param(std::string_view name, ParamSource source,
                          const int *symbol = nullptr) {
        m_params.emplace_back(name, static_cast<int *>(nullptr), 1,
                              lookup_symbol(symbol), nullptr, source, -1);
    }

    // --- Per call ---

    // The application's and the call's values, at invoke.
    void update(op_arg *args, int nargs) {
        for (auto& param : m_params)
            param.update(args, nargs);
    }

    // The values update gave, for a queued call, and back before it runs.
    std::vector<std::byte> save() const {
        std::vector<std::byte> out;
        for (const auto& param : m_params)
            param.save(out);
        return out;
    }

    void restore(const std::vector<std::byte>& in) {
        const std::byte *cursor = in.data();
        for (auto& param : m_params)
            cursor = param.restore(cursor);
        assert(cursor == in.data() + in.size());
    }

    // Upload the changed values an offline kernel reads, under a strategy's
    // launch values.
    void upload(const LaunchValues& launch) {
        for (auto& param : m_params) {
            param.apply(launch);
            param.upload();
        }
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
    // compile starting once asked OP_JIT_MIN_CALLS times, at once for
    // Urgency::now, and joined for Urgency::wait when no offline kernel can
    // run meanwhile.  pending while nothing can run, unavailable when
    // nothing ever can.
    Acquired acquire(JitKernels& kernels, const KernelSource& source,
                     const LaunchValues& launch, op::loop::Urgency urgency) {
        const auto& run = config();
        bool offline = offline_usable(kernels, source, launch.block_size);
        if (!run.enable)
            return {offline ? op::loop::Status::ready : op::loop::Status::unavailable,
                    nullptr};
        bool specialise = !offline || run.force ||
                          kernels.m_offline_attrs.numRegs > run.min_registers;
        if (!specialise)
            return {op::loop::Status::ready, nullptr};

        auto hash = hash_params(launch);
        // Waiting is for a strategy with nothing else to run.
        bool wait = urgency == op::loop::Urgency::wait && !offline;
        auto acquired = acquire_compiled(
            kernels, m_headers, source.name, hash, urgency, wait,
            [&] {
                return preamble(m_headers) + format_params() +
                       std::string{source.decls} + std::string{source.source};
            },
            m_loop);
        if (acquired.kernel == nullptr && offline)
            acquired.status = op::loop::Status::ready;
        return acquired;
    }

};

} // namespace op::gpu
