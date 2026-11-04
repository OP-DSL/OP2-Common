#pragma once

// The CUDA and HIP backend's executor, made with its loop from the generated
// code's descriptor (op2/rt/abi/gpu/descriptor.h): the parameters' device
// symbols, the compiler of its specialised kernels, the launches' arguments
// and events, and the call's device globals.  Defined in
// src/rt/gpu/op_gpu_executor.cpp.

#include <op2/gpu/shims.h>
#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/abi/gpu/launch.h>
#include <op2/rt/gpu/jit.h>
#include <op2/rt/gpu/kernel.h>
#include <op2/rt/jit/params.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/ladder.h>
#include <op2/rt/loop/loop.h>
#include <op_lib_core.h>

#include <cstddef>
#include <span>
#include <vector>

namespace op::rt::gpu {

class Executor final : public op::rt::loop::Executor {
private:
    const abi::gpu::Loop& m_descriptor;
    op::rt::jit::ParamSet& m_params;
    op::rt::Context& m_context;
    SymbolBindings m_symbols;
    Compiler m_compiler;
    // Per argument, whether every kernel has it by value; and the call's
    // flags, which add the READ globals the chosen kernel bakes in.
    std::vector<unsigned char> m_by_value;
    std::vector<unsigned char> m_on_host;
    // Event pairs bracketing each phase's launches of a timed call.
    std::vector<gpuEvent_t> m_events;
    // A launch's kernel arguments: its record's, then the loop's.
    std::vector<void *> m_arguments;
    // Whether a call has needed more than 8 KiB of reduction scratch a thread.
    mutable bool m_warned_reductions = false;

    // Register a parameter, bound to the device copy an offline kernel reads
    // it from when it has one.
    void add_param(const abi::gpu::Param& param);

    // Per argument, whether the kernel that runs has it from the host side
    // and so needs no device copy of it.
    const unsigned char *on_host(const op::rt::loop::Prepared& prepared, int nargs);

public:
    // The executor of descriptor's loop in context, its parameters registered
    // in params.
    Executor(const abi::gpu::Loop& descriptor, op::rt::jit::ParamSet& params,
             op::rt::Context& context);
    ~Executor() override;

    void add_strategies(op::rt::loop::Ladder& ladder) override;

    // The data lives on the device.
    int exchange_halos(op_set set, std::span<op_arg> args) override;
    void mark_dirty(std::span<op_arg> args) override;

    // --- For the strategies ---

    const char *name() const { return m_descriptor.name; }
    bool direct() const { return m_descriptor.direct; }
    op::rt::jit::ParamSet& params() { return m_params; }
    op::rt::Context& context() { return m_context; }
    const Compiler& compiler() const { return m_compiler; }
    const abi::gpu::OfflineCalls *calls() const { return m_descriptor.calls; }

    // Upload the changed values an offline kernel reads from device symbols.
    void upload_symbols() { m_symbols.upload(m_params); }

    // The most blocks of block_size threads a launch of call may use: as
    // many as the reduction scratch allows.  A call needing more than 8 KiB a
    // thread is warned about once.
    int block_limit(const op::rt::loop::Call& call, int block_size) const;

    // The call's device copies of the globals, global_stride apart as the
    // selected strategy launches its blocks; those its kernel has from the
    // host need none.
    void prepare_globals(const op::rt::loop::Prepared& prepared, const op::rt::loop::Call& call,
                         int global_stride);

    // Combine the call's device copies of the globals, global_stride apart,
    // into its arguments; whether the host must synchronise before reading
    // them.
    bool combine_globals(const op::rt::loop::Call& call, int global_stride);

    // A launch's kernel arguments: its record's fields, then the loop's
    // arguments, the active optional ones marked in the record.
    void **arguments(abi::gpu::Range& record, std::span<op_arg> args);
    void **arguments(abi::gpu::Chunk& record, std::span<op_arg> args);

    gpuEvent_t event(std::size_t index);
};

} // namespace op::rt::gpu
