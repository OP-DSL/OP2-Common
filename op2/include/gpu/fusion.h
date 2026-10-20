#pragma once

// Runtime kernel fusion for the CUDA and HIP backend.  When the lazy queue
// runs, consecutive calls over one set whose dependencies are element-wise
// share one JIT kernel: for each element it runs each call's op2_body in
// queue order.  Each call's strategy supplies its declarations and its
// element function (FusionSource); only the wrapper is written here.

#include <gpu/jit.h>
#include <gpu/lazy.h>
#include <gpu/strategy.h>
#include <op_lib_core.h>
#include <op_profile.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace op::gpu {

// A queued call as a member of a fused kernel, from its loop's preferred
// strategy.
struct FusionMember {
    PendingCall *call = nullptr;
    StrategyBase *strategy = nullptr;
    // The elements, in one launch, and the arrays' stride.
    int begin = 0;
    int end = 0;
    int set_stride = 0;
    int block_size = 0;
    // The most blocks the reduction scratch allows, the blocks the call
    // would launch, and its per-thread global stride.
    int block_limit = 0;
    int max_blocks = 0;
    int global_stride = 0;
    // The loop, strategy and JIT parameter values, hashed.
    std::uint64_t key = 0;
    // Whether the call's globals take device storage, which the calls of a
    // kernel share.
    bool device_globals = false;
    bool global_output = false;
    // The strategy's offline kernel's registers, 0 when unknown.
    int registers = 0;
};

namespace fusion_detail {

inline bool writes(const op_arg& arg) {
    return arg.acc == OP_WRITE || arg.acc == OP_RW || arg.acc == OP_INC;
}

// How a call uses one dat.
struct Use {
    op_dat dat = nullptr;
    bool read = false;
    bool write = false;
    bool only_inc = true;
    bool all_direct = true;
    bool indirect_inc = false;
};

inline std::vector<Use> uses(const PendingCall& call) {
    std::vector<Use> out;
    for (const auto& arg : call.args) {
        if (arg.opt == 0 || arg.argtype != OP_ARG_DAT)
            continue;
        auto found = std::find_if(out.begin(), out.end(),
                                  [&](const Use& use) { return use.dat == arg.dat; });
        if (found == out.end()) {
            out.push_back({arg.dat});
            found = out.end() - 1;
        }
        bool direct = arg.map == nullptr;
        found->read |= arg.acc == OP_READ || arg.acc == OP_RW;
        found->write |= writes(arg);
        found->only_inc &= arg.acc == OP_INC;
        found->all_direct &= direct;
        found->indirect_inc |= !direct && arg.acc == OP_INC;
    }
    return out;
}

// Whether b may run after a for each element in turn: every dat either
// writes is reached only directly by both, or only incremented by both
// with every indirect increment atomic.
inline bool element_wise(const std::vector<Use>& a, bool a_atomic,
                         const std::vector<Use>& b, bool b_atomic) {
    for (const auto& x : a)
        for (const auto& y : b) {
            if (x.dat != y.dat || !(x.write || y.write))
                continue;
            if (x.all_direct && y.all_direct)
                continue;
            if (x.only_inc && y.only_inc && (!x.indirect_inc || a_atomic) &&
                (!y.indirect_inc || b_atomic))
                continue;
            return false;
        }
    return true;
}

inline bool atomic(const FusionMember& member) {
    return member.strategy->family() == "atomics";
}

} // namespace fusion_detail

class Fusion {
private:
    struct Entry {
        std::string name;
        bool checked = false;
        bool rejected = false;
        int registers = 0;
    };

    JitKernels m_kernels;
    std::unordered_map<std::uint64_t, Entry> m_entries;
    std::size_t m_fused_launches = 0;
    std::size_t m_fused_calls = 0;

    static void release_callback(void *owner) {
        static_cast<Fusion *>(owner)->release();
    }

    void release() {
        m_kernels.join();
        if (OP_diags > 1 && m_fused_launches > 0)
            op_printf("fusion: %zu fused launches ran %zu calls\n", m_fused_launches,
                      m_fused_calls);
        m_fused_launches = 0;
        m_fused_calls = 0;
    }

    static QueuedLoop& loop_of(const FusionMember& member) { return *member.call->loop; }

    static bool fusable(const FusionMember& member) {
        return config().fusion_max_registers <= 0 || member.registers == 0 ||
               member.registers <= config().fusion_max_registers;
    }

    // Whether member can join the chain.
    static bool joins(const std::vector<FusionMember>& chain,
                      const FusionMember& member) {
        const auto& first = chain.front();
        if (member.call->set != first.call->set ||
            member.block_size != first.block_size || member.begin != first.begin ||
            member.end != first.end)
            return false;
        int device_globals = member.device_globals;
        auto uses = fusion_detail::uses(*member.call);
        for (const auto& other : chain) {
            device_globals += other.device_globals;
            if (!fusion_detail::element_wise(fusion_detail::uses(*other.call),
                                             fusion_detail::atomic(other), uses,
                                             fusion_detail::atomic(member)))
                return false;
        }
        return device_globals <= 1;
    }

    // A fused kernel's arguments: the distinct dat and map pointers in
    // first-use order with each global's own, each member's pointer
    // parameters as indices into them, and its values.
    struct Arguments {
        std::vector<void *> buffers;
        std::vector<int> buffer_index;
        std::vector<int> values;
        // Kept alive until the launch: the bindings point into them.
        std::vector<LaunchContext> contexts;
    };

    static Arguments bind(std::vector<FusionMember>& chain) {
        Arguments arguments;
        arguments.contexts.resize(chain.size());
        std::vector<void *> offline, jit;
        for (std::size_t k = 0; k < chain.size(); ++k) {
            auto& member = chain[k];
            auto& context = arguments.contexts[k];
            context.global_stride = member.global_stride;
            context.start = member.begin;
            context.end = member.end;
            context.set_stride = member.set_stride;
            member.strategy->binder()(context, member.call->args.data(), offline, jit);
            const auto& params = member.strategy->source().fusion->params;
            assert(jit.size() == params.size());
            for (std::size_t i = 0; i < params.size(); ++i) {
                switch (params[i].kind) {
                case FusionParam::buffer: {
                    void *pointer = *static_cast<void *const *>(jit[i]);
                    auto found = std::find(arguments.buffers.begin(),
                                           arguments.buffers.end(), pointer);
                    arguments.buffer_index.push_back(
                        static_cast<int>(found - arguments.buffers.begin()));
                    if (found == arguments.buffers.end())
                        arguments.buffers.push_back(pointer);
                    break;
                }
                case FusionParam::global:
                    // Placed by the globals' preparation, so never shared.
                    arguments.buffer_index.push_back(
                        static_cast<int>(arguments.buffers.size()));
                    arguments.buffers.push_back(*static_cast<void *const *>(jit[i]));
                    break;
                case FusionParam::value: {
                    int value;
                    std::memcpy(&value, jit[i], sizeof(int));
                    arguments.values.push_back(value);
                    break;
                }
                default:
                    break;
                }
            }
        }
        return arguments;
    }

    static std::uint64_t chain_hash(const std::vector<FusionMember>& chain,
                                    const Arguments& arguments) {
        std::uint64_t hash = op::gpu::hash(chain.front().block_size);
        for (const auto& member : chain)
            hash = op::gpu::hash(member.key, hash);
        // Which parameters share a pointer shapes the kernel.
        return hash_bytes(arguments.buffer_index.data(),
                          arguments.buffer_index.size() * sizeof(int), hash);
    }

    static std::string chain_name(const std::vector<FusionMember>& chain) {
        std::string name;
        for (const auto& member : chain) {
            name += name.empty() ? "" : "+";
            name += loop_of(member).fusion_name();
        }
        return name;
    }

    // The fused kernel's source: each member's constants and declarations
    // in a namespace of its own, so two members never collide, then the
    // wrapper.  Its pointer parameters carry no __restrict, as members may
    // pass one dat twice; each op2_body keeps its own.
    static std::string source(std::vector<FusionMember>& chain,
                              const Arguments& arguments) {
        std::string text = JitService::preamble(loop_of(chain.front()).fusion_headers());
        for (std::size_t k = 0; k < chain.size(); ++k) {
            auto& member = chain[k];
            text += "namespace op2_f" + std::to_string(k) + " {\n";
            text += loop_of(member).fusion_params(member);
            text += member.strategy->source().decls;
            text += "}\n";
        }

        std::string signature;
        for (std::size_t b = 0; b < arguments.buffers.size(); ++b)
            signature += "    void *b" + std::to_string(b) + ",\n";
        std::string body;
        std::size_t buffer = 0, value = 0;
        for (std::size_t k = 0; k < chain.size(); ++k) {
            const auto& fusion = *chain[k].strategy->source().fusion;
            body += "        op2_f" + std::to_string(k) + "::" + fusion.body + "(";
            for (const auto& param : fusion.params) {
                if (param.kind == FusionParam::buffer || param.kind == FusionParam::global) {
                    body += "(" + std::string{param.type} + ")b" +
                            std::to_string(arguments.buffer_index[buffer++]) + ", ";
                } else if (param.kind == FusionParam::value) {
                    std::string name = "v" + std::to_string(value++);
                    signature += "    " + std::string{param.type} + " " + name + ",\n";
                    body += name + ", ";
                }
            }
            body += "thread_id, n);\n";
        }

        text += "\nextern \"C\" __global__\n";
#ifdef OP2_HIP
        text += "__launch_bounds__(" + std::to_string(chain.front().block_size) + ")\n";
#endif
        text += "void op2_fused(\n" + signature +
                "    const int start,\n    const int end\n) {\n"
                "    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;\n"
                "    for (int n = thread_id + start; n < end; n += blockDim.x * gridDim.x) {\n" +
                body + "    }\n}\n";
        return text;
    }

    // The chain's kernel once compiled; starts its compile as the JIT
    // policy says.
    JitKernel *kernel(std::uint64_t hash, std::vector<FusionMember>& chain,
                      const Arguments& arguments) {
        auto& entry = m_entries[hash];
        if (entry.rejected)
            return nullptr;
        if (entry.name.empty())
            entry.name = chain_name(chain);

        auto acquired = JitService::acquire_compiled(
            m_kernels, loop_of(chain.front()).fusion_headers(), "op2_fused", hash,
            op::loop::Urgency::later, false, [&] { return source(chain, arguments); },
            entry.name);
        auto *compiled = acquired.kernel;
        if (compiled == nullptr || entry.checked)
            return compiled;

        // A fused kernel needing more registers than any member is rejected
        // when that costs occupancy.
        entry.checked = true;
        entry.registers = compiled->registers();
        int member_occupancy = INT32_MAX;
        for (const auto& member : chain)
            if (member.registers > 0)
                member_occupancy = std::min(
                    member_occupancy, config().occupancy(member.registers, member.block_size));
        int fused_occupancy = compiled->occupancy(chain.front().block_size);
        entry.rejected = member_occupancy != INT32_MAX && fused_occupancy < member_occupancy;
        if (OP_diags > 2)
            op_printf("fusion: %s uses %d registers, %d blocks per SM (members %d)%s\n",
                      entry.name.c_str(), entry.registers, fused_occupancy,
                      member_occupancy == INT32_MAX ? 0 : member_occupancy,
                      entry.rejected ? ", rejected" : "");
        return entry.rejected ? nullptr : compiled;
    }

    void launch(std::vector<FusionMember>& chain) {
        if (chain.size() == 1) {
            chain.front().call->loop->execute(*chain.front().call);
            return;
        }
        auto arguments = bind(chain);
        auto hash = chain_hash(chain, arguments);
        auto *compiled = kernel(hash, chain, arguments);
        if (compiled == nullptr) {
            for (auto& member : chain)
                member.call->loop->execute(*member.call);
            return;
        }

        const auto& name = m_entries[hash].name;
        op_profile_enter_kernel(name.c_str(), "c_CUDA", "Fused");
        op_profile_enter("Computation");
        for (auto& member : chain)
            loop_of(member).fused_begin(member);
        // After the exchanges and the globals' preparation, which may move
        // device pointers.
        arguments = bind(chain);

        const auto& first = chain.front();
        int blocks = (first.end - first.begin + first.block_size - 1) / first.block_size;
        for (const auto& member : chain)
            blocks = std::min({blocks, member.block_limit, member.max_blocks});
        if (blocks > 0) {
            std::vector<void *> argv;
            for (auto& buffer : arguments.buffers)
                argv.push_back(&buffer);
            for (auto& value : arguments.values)
                argv.push_back(&value);
            int start = first.begin, end = first.end;
            argv.push_back(&start);
            argv.push_back(&end);
            compiled->launch(blocks, first.block_size, argv.data(), 0);
        }

        for (auto& member : chain)
            loop_of(member).fused_end(member);
        op_profile_exit();
        op_profile_exit();
        ++m_fused_launches;
        m_fused_calls += chain.size();
    }

public:
    Fusion() { op::loop::register_release(this, &Fusion::release_callback); }
    ~Fusion() { op::loop::unregister_release(this); }
    Fusion(const Fusion&) = delete;

    // Run the queued calls in order, as fused kernels where they may share
    // one.
    void run(std::vector<PendingCall>& calls) {
        std::vector<FusionMember> chain;
        auto close = [&] {
            if (!chain.empty())
                launch(chain);
            chain.clear();
        };
        for (auto& call : calls) {
            FusionMember member;
            bool eligible = config().fusion &&
                            call.loop->fusion_member(call, member) &&
                            fusable(member);
            if (!eligible) {
                close();
                call.loop->execute(call);
                continue;
            }
            if (!chain.empty() && !joins(chain, member))
                close();
            chain.push_back(member);
        }
        close();
    }
};

inline Fusion& fusion() {
    static Fusion instance;
    return instance;
}

} // namespace op::gpu
