#pragma once

// The CUDA and HIP backend's loop, built from the generated code's
// descriptor (op2/rt/abi/gpu/descriptor.h) on its first call: the base loop,
// plus the parameters' device symbols, the compiler of its specialised
// kernels, the launches' arguments and the call's device globals.  Defined
// in src/rt/gpu/op_gpu_loop.cpp.

#include <op2/gpu/shims.h>
#include <op2/rt/abi/descriptor.h>
#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/abi/gpu/launch.h>
#include <op2/rt/loop/call.h>
#include <op2/rt/loop/loop.h>
#include <op_lib_core.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace op::rt::gpu {

class Compiler;
class SymbolBindings;

class Loop : public op::rt::loop::Loop {
private:
    bool m_gbl_inc_atomic;
    abi::gpu::Binder m_bind;
    const abi::gpu::OfflineCalls *m_calls;
    std::unique_ptr<SymbolBindings> m_symbols;
    std::unique_ptr<Compiler> m_compiler;
    // Per argument, whether every kernel has it by value; and the call's
    // flags, which add the READ globals the chosen kernel bakes in.
    std::vector<unsigned char> m_by_value;
    std::vector<unsigned char> m_on_host;
    // Event pairs bracketing each phase's launches of a timed call.
    std::vector<gpuEvent_t> m_events;
    // A launch's kernel arguments: its record's, then the loop's.
    std::vector<void *> m_arguments;
    // The call under way: its strategy's stride between the blocks' global
    // copies, and whether their processing needs a synchronisation before
    // the host reads.
    int m_global_stride = 0;
    bool m_exit_sync = false;

    // Register a parameter, bound to the device copy an offline kernel reads
    // it from when it has one.
    void add_param(const abi::gpu::Param& param);

    // Per argument, whether the kernel that runs has it from the host side
    // and so needs no device copy of it.
    const unsigned char *on_host(const op::rt::loop::Prepared& prepared, int nargs);

public:
    explicit Loop(const abi::gpu::Loop& descriptor);
    ~Loop() override;

    // --- For the strategies ---

    const Compiler& compiler() const { return *m_compiler; }
    const abi::gpu::OfflineCalls *calls() const { return m_calls; }

    // Upload the changed values an offline kernel reads from device symbols.
    void upload();

    // The most blocks a launch may use: as many as the reduction scratch
    // allows.
    int block_limit(op_arg *args, int nargs, int block_size) const;

    // The call's device copies of the globals, global_stride apart as the
    // selected strategy launches its blocks; those its kernel has from the
    // host need none.
    void prepare_globals(const op::rt::loop::Prepared& prepared, op_arg *args, int nargs,
                         int global_stride);

    // A launch's kernel arguments: its record's fields, then the loop's
    // arguments, the active optional ones marked in the record.
    void **arguments(abi::gpu::Range& record, op_arg *args);
    void **arguments(abi::gpu::Chunk& record, op_arg *args);

    gpuEvent_t event(std::size_t index);

protected:
    // --- Per call ---

    // The data lives on the device.
    int exchange_halos(op_set set, op_arg *args, int nargs) override;
    void mark_dirty(op_arg *args, int nargs) override;

    void combine_globals(op_arg *args, int nargs) override;

    void end_call() override;
};

} // namespace op::rt::gpu
