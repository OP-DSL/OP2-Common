#pragma once

// The CUDA and HIP backend as op2_kernels sees it: the loop the generated
// code builds per par_loop and registers its kernels, bindings and
// parameters on, and what a binding and an offline kernel's calls
// (op2/rt/kernel/gpu/launch.h) are given.  The loop adds to the base loop the
// parameters' device symbols, the compiler of its specialised kernels, the
// launches and the call's device globals; it and its strategies are defined
// in src/rt/gpu.

#include <op2/gpu/shims.h>
#include <op2/rt/loop/loop.h>
#include <op2/rt/loop/strategy.h>
#include <op2/rt/plan/staging.h>
#include <op_lib_core.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::gpu {

class Compiler;
class SymbolBindings;
struct LaunchValues;

// The block size a strategy runs unless OP_BLOCK_SIZES says otherwise, and
// the offline kernels' launch bounds; a specialised kernel has its own block
// size as launch bounds.
constexpr int default_block_size = 128;

// A chunk plan's arrays on the device, as a staged kernel reads them.
struct ChunkArrays {
    int *source_offsets = nullptr;
    op::rt::plan::StageWord *stage_words = nullptr;
    // Each region's slot count, from the plan's offsets: a kernel reads its
    // counts up front and its offsets only when it flushes, so no offset
    // stays in a register through the user kernel.
    int *stage_counts = nullptr;
    int *slot_offsets = nullptr;
    int *slot_targets = nullptr;
    int *chunk_order = nullptr;
    std::uint8_t *thread_colours = nullptr;
    int *chunk_thread_colours = nullptr;

    // Wavefront plans: predecessors, a done flag per position holding the
    // epoch of the call that last ran it, and a counter per section.
    int *wave_pred_offsets = nullptr;
    int *wave_preds = nullptr;
    int *wave_done = nullptr;
    int *wave_tickets = nullptr;
};

// What the generated bindings bind a launch's kernel arguments to.
struct LaunchContext {
    int global_stride = 0;
    unsigned opt_flags = 0;
    int start = 0;
    int end = 0;
    int set_stride = 0;
    // A set colouring's elements, colour by colour.
    const int *colour_order = nullptr;

    struct {
        const ChunkArrays *plan = nullptr;
        int chunk_begin = 0;
        int chunk_end = 0;
        // This section's position counter under a wavefront partition.
        int *wave_ticket = nullptr;
        int wave_epoch = 0;
    } chunk;
};

// Fill a launch's kernel arguments, one list for its offline and JIT
// kernels alike, from the context.
using Binder = void (*)(LaunchContext& launch, op_arg *args,
                        std::vector<void *>& arguments);

// A kernel's resources as its compiler left them: registers and local
// memory per thread, static shared memory per block, and the most threads a
// block can have with them.
struct KernelFigures {
    int registers = 0;
    int local_bytes = 0;
    int shared_bytes = 0;
    int max_threads = 0;

    // "64 regs", with the local and shared memory when there is any.
    std::string text() const;
};

// The calls on a kernel compiled with the application, which op2_kernels
// makes with the runtime that registered it.
struct OfflineCalls {
    KernelFigures (*figures)(const void *function);
    // Let a launch have up to bytes of dynamic shared memory.
    void (*allow_shared)(const void *function, int bytes);
    void (*launch)(const void *function, int blocks, int block_size, void **args,
                   int shared_bytes);
};

// A kernel compiled with the application, if any, and its calls.
struct OfflineFunction {
    const void *function = nullptr;
    const OfflineCalls *calls = nullptr;
};

// What a strategy's kernel is built from: the offline function, if any, and
// the source a specialised one is compiled from.
struct KernelSource {
    std::string_view name;
    OfflineFunction offline;
    std::string_view source;
};

// Each value a strategy fixes for its kernel, by its slot.
enum class LaunchValue {
    global_stride,
    block_size,
    unstaged_singles,
    wavefront,
};

// The headers a compile includes, embedded in the translation unit.
struct JitHeaders {
    std::vector<const char *> data;
    std::vector<const char *> names;
    // Whether op_f2c_params.h, the application's constants, is among them.
    bool params = false;
};

class Loop : public op::rt::loop::Loop {
private:
    // A global OP_INC is incremented atomically, in place, rather than
    // reduced from a copy per thread.
    bool m_gbl_inc_atomic;
    std::unique_ptr<SymbolBindings> m_symbols;
    std::unique_ptr<Compiler> m_compiler;
    // Per argument, whether every kernel has it by value; and the call's
    // flags, which add the READ globals the chosen kernel bakes in.
    std::vector<unsigned char> m_by_value;
    std::vector<unsigned char> m_on_host;
    // Event pairs bracketing each phase's launches of a timed call.
    std::vector<gpuEvent_t> m_events;
    // A launch's kernel arguments, filled by the binders.
    std::vector<void *> m_arguments;
    // The call under way: its strategy's stride between the blocks' global
    // copies, and whether their processing needs a synchronisation before
    // the host reads.
    int m_global_stride = 0;
    bool m_exit_sync = false;

    // Bind a parameter to the device symbol an offline kernel reads it from,
    // if there is one.
    void bind(std::size_t param, void *symbol_d, std::uint64_t *shared_hash);

    void mark_by_value(int arg_index);

    // Per argument, whether the kernel that runs has it from the host side
    // and so needs no device copy of it.
    const unsigned char *on_host(const op::rt::loop::Prepared& prepared, int nargs);

public:
    // profile_target and profile_variant name the loop to the profiler.
    Loop(std::string_view name, std::string_view profile_target,
         std::string_view profile_variant, bool direct, bool gbl_inc_atomic,
         JitHeaders headers);
    ~Loop() override;

    // --- Registration, by the generated code ---

    // Add a family, its strategies running source's kernels with arguments
    // from bind; the chunked families' groups are the arguments they stage.
    void add(std::string_view family, KernelSource source, Binder bind,
             op::rt::plan::ArgGroups groups = {});

    // The parameters the kernels are specialised on.  symbol_d is the device
    // address of the copy an offline kernel reads, if any
    // (op::rt::kernel::gpu::symbol), and hash_device_ptr the hash of its
    // value there, shared by every loop that reads it.  A built kernel can
    // have each at run time but a dat's dimension and a strategy's value,
    // which its source needs as constants.

    template<typename T>
    void add_param(std::string_view name, T *data, void *symbol_d = nullptr,
                   std::uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add(name, data, op::rt::loop::ParamSource::external, -1, true),
             symbol_d, hash_device_ptr);
    }

    template<typename T>
    void add_param(std::string_view name, T *data, std::size_t len,
                   void *symbol_d = nullptr, std::uint64_t *hash_device_ptr = nullptr) {
        bind(m_params.add_array(name, data, len, op::rt::loop::ParamSource::external, -1,
                                true),
             symbol_d, hash_device_ptr);
    }

    // A scalar READ global, which every kernel has by value.
    template<typename T>
    void add_scalar_arg_param(std::string_view name, T *data, int arg_index) {
        m_params.add(name, data, op::rt::loop::ParamSource::scalar_arg, arg_index, true);
        mark_by_value(arg_index);
    }

    // Any other READ global: by value, or else read by an offline kernel
    // from its device copy.
    template<typename T>
    void add_global_param(std::string_view name, int arg_index, bool by_value = false) {
        m_params.add_array(name, static_cast<T *>(nullptr), 0,
                           op::rt::loop::ParamSource::array_arg, arg_index, true);
        if (by_value)
            mark_by_value(arg_index);
    }

    void add_dat_stride_param(std::string_view name, int *data, int arg_index,
                              void *symbol_d);

    void add_dat_dim_param(std::string_view name, int *data, int arg_index);

    // A value each strategy decides, given to every acquire and upload.
    void add_launch_param(std::string_view name, LaunchValue value,
                          void *symbol_d = nullptr);

    // --- For the strategies ---

    const Compiler& compiler() const { return *m_compiler; }

    // Upload the changed values an offline kernel reads, under a strategy's
    // launch values.
    void upload(const LaunchValues& launch);

    // The most blocks a launch may use: as many as the reduction scratch
    // allows.
    int block_limit(op_arg *args, int nargs, int block_size) const;

    std::vector<void *>& arguments() { return m_arguments; }

    gpuEvent_t event(std::size_t index);

    // At op_exit, while the device is still up, once the strategies are
    // released: free the timing events.
    void release() override;

protected:
    // --- Per call ---

    // The device copies of the globals, the strategy's stride apart.
    void begin_call(const op::rt::loop::Call& call,
                    const op::rt::loop::Autotuner::Decision& decision, op_arg *args,
                    int nargs) override;

    void combine_globals(op_arg *args, int nargs) override;

    void end_call() override;
};

} // namespace op::rt::gpu
