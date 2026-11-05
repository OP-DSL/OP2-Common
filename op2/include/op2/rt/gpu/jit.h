#pragma once

// The CUDA and HIP JIT mechanism: kernels built through NVRTC or hipRTC, and
// how they have their run-time parameters.  The policy, the parameters and
// the build cache are op2/rt/jit's, the built kernels op2/rt/gpu/kernel.h's.
// Defined in src/rt/gpu/op_gpu_jit.cpp.

#include <op2/rt/abi/gpu/descriptor.h>
#include <op2/rt/gpu/kernel.h>
#include <op2/rt/jit/params.h>

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace op::rt::gpu {

// --- Builds -------------------------------------------------------------------------

// Builds a loop's specialised kernels through NVRTC or hipRTC.
class Compiler {
private:
    std::string m_loop;
    const char *m_translated;
    abi::gpu::JitHeaders m_headers;
    std::string m_arch;

public:
    // A build of one kernel, which returns it, or none when the compile
    // failed after printing the compiler's log.  It holds what it reads, so
    // its loop's compiler may go before the build finishes.
    struct Job {
        std::string loop;
        abi::gpu::JitHeaders headers;
        std::string arch;
        std::string text;
        std::string name;
        std::vector<RuntimeSymbol> symbols;
        std::vector<int> baked_globals;

        std::unique_ptr<JitKernel> operator()();
    };

    // The compiler of loop's kernels, each family's source after translated,
    // from headers, for arch, NVRTC's architecture flag (empty under HIP).
    Compiler(std::string_view loop, const char *translated, abi::gpu::JitHeaders headers,
             std::string_view arch)
        : m_loop{loop}, m_translated{translated}, m_headers{headers}, m_arch{arch} {}

    // The job that builds family's kernel with the parameters' current values
    // and the strategy's, by slot, baked in, or at run time, as modes say.
    Job job(const op::rt::jit::ParamSet& params, std::span<const op::rt::jit::ParamMode> modes,
            std::span<const int> values, const abi::gpu::Family& family) const;
};

} // namespace op::rt::gpu
