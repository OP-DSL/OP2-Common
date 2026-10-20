#pragma once

// Runtime kernel fusion for the CUDA and HIP backend.  When the lazy queue
// runs, its calls are grouped into fused kernels.  A chain is calls over one
// set whose dependencies are element-wise: for each element the kernel runs
// each call's op2_body in queue order.  Independent chains share a kernel in
// disjoint block ranges.  Each call's strategy supplies its declarations and
// its element function (FusionSource); only the wrapper is written here.

#include <gpu/jit.h>
#include <gpu/lazy.h>
#include <gpu/strategy.h>
#include <loop/fusion.h>
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


class Fusion {
private:
    using Chain = std::vector<FusionMember>;
    using Group = std::vector<Chain>;

    struct Entry {
        std::string name;
        bool checked = false;
        bool rejected = false;
        int registers = 0;
        // Timed runs, fused and apart, alternating from fused; decided once
        // each has its warm-up and samples.
        std::vector<float> fused_times;
        std::vector<float> apart_times;
        bool decided = false;
    };

    static constexpr std::size_t tune_samples = 3;

    JitKernels m_kernels;
    gpuEvent_t m_events[2] = {};
    std::unordered_map<std::uint64_t, Entry> m_entries;
    std::size_t m_fused_launches = 0;
    std::size_t m_fused_calls = 0;

    static void release_callback(void *owner) {
        static_cast<Fusion *>(owner)->release();
    }

    void release() {
        m_kernels.join();
        for (auto& event : m_events)
            if (event != nullptr) {
                gpuEventDestroy(event);
                event = nullptr;
            }
        if (OP_diags > 1 && m_fused_launches > 0)
            op_printf("fusion: %zu fused launches ran %zu calls\n", m_fused_launches,
                      m_fused_calls);
        m_fused_launches = 0;
        m_fused_calls = 0;
    }

    static QueuedLoop& loop_of(const FusionMember& member) { return *member.call->loop; }

    static std::vector<op::loop::fusion::Use> uses(const FusionMember& member) {
        return op::loop::fusion::uses(member.call->args);
    }

    // Whether the member's indirect increments are atomic.
    static bool atomic(const FusionMember& member) {
        return member.strategy->family() == "atomics";
    }

    static bool fusable(const FusionMember& member) {
        return config().fusion_max_registers <= 0 || member.registers == 0 ||
               member.registers <= config().fusion_max_registers;
    }

    template<typename F>
    static void for_members(Group& group, F&& f) {
        for (auto& chain : group)
            for (auto& member : chain)
                f(member);
    }

    static std::size_t members(const Group& group) {
        std::size_t count = 0;
        for (const auto& chain : group)
            count += chain.size();
        return count;
    }

    // Whether member may run after every call of chain, element by element.
    static bool joins(const Chain& chain, const FusionMember& member,
                      const std::vector<op::loop::fusion::Use>& member_uses) {
        const auto& first = chain.front();
        if (member.call->set != first.call->set || member.begin != first.begin ||
            member.end != first.end)
            return false;
        for (const auto& other : chain)
            if (!op::loop::fusion::element_wise(uses(other), atomic(other), member_uses,
                                                atomic(member)))
                return false;
        return true;
    }

    static bool independent(const Chain& chain, const FusionMember& member,
                            const std::vector<op::loop::fusion::Use>& member_uses) {
        for (const auto& other : chain)
            if (!op::loop::fusion::independent(uses(other), atomic(other), member_uses,
                                               atomic(member)))
                return false;
        return true;
    }

    // Place member in the group: in the first chain it may follow, being
    // independent of the others, else in a chain of its own, being
    // independent of all.  False when it fits nowhere.
    static bool place(Group& group, const FusionMember& member) {
        if (group.empty()) {
            group.push_back({member});
            return true;
        }
        if (member.block_size != group.front().front().block_size)
            return false;
        int device_globals = member.device_globals;
        for_members(group, [&](const FusionMember& other) {
            device_globals += other.device_globals;
        });
        if (device_globals > 1)
            return false;

        auto member_uses = uses(member);
        std::vector<bool> apart(group.size());
        for (std::size_t c = 0; c < group.size(); ++c)
            apart[c] = independent(group[c], member, member_uses);
        for (std::size_t c = 0; c < group.size(); ++c) {
            if (!joins(group[c], member, member_uses))
                continue;
            bool others = true;
            for (std::size_t d = 0; d < group.size(); ++d)
                others = others && (d == c || apart[d]);
            if (others) {
                group[c].push_back(member);
                return true;
            }
        }
        if (std::all_of(apart.begin(), apart.end(), [](bool b) { return b; })) {
            group.push_back({member});
            return true;
        }
        return false;
    }

    // A fused kernel's arguments: the distinct dat and map pointers in
    // first-use order with each global's own, each member's pointer
    // parameters as indices into them, and its values.
    struct Arguments {
        std::vector<void *> buffers;
        std::vector<int> buffer_index;
        std::vector<int> values;
        // The scalar READ globals' values, each in a slot of its own.
        std::vector<std::uint64_t> scalars;
        // Kept alive until the launch: the bindings point into them.
        std::vector<LaunchContext> contexts;
    };

    static Arguments bind(Group& group) {
        Arguments arguments;
        arguments.contexts.reserve(members(group));
        std::vector<void *> offline, jit;
        for_members(group, [&](FusionMember& member) {
            auto& context = arguments.contexts.emplace_back();
            context.global_stride = member.global_stride;
            context.start = member.begin;
            context.end = member.end;
            context.set_stride = member.set_stride;
            member.strategy->binder()(context, member.call->args.data(), offline, jit);
            const auto& params = member.strategy->source().fusion->params;
            std::size_t slot = 0;
            for (const auto& param : params) {
                if (param.kind == FusionParam::scalar) {
                    const auto& arg = member.call->args[static_cast<std::size_t>(param.arg)];
                    std::uint64_t value = 0;
                    std::memcpy(&value, arg.data, std::min<std::size_t>(arg.size, sizeof(value)));
                    arguments.scalars.push_back(value);
                    continue;
                }
                void *bound = jit[slot++];
                switch (param.kind) {
                case FusionParam::buffer: {
                    void *pointer = *static_cast<void *const *>(bound);
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
                    arguments.buffers.push_back(*static_cast<void *const *>(bound));
                    break;
                case FusionParam::value: {
                    int value;
                    std::memcpy(&value, bound, sizeof(int));
                    arguments.values.push_back(value);
                    break;
                }
                default:
                    break;
                }
            }
            assert(slot == jit.size());
        });
        return arguments;
    }

    static std::uint64_t group_hash(const Group& group, const Arguments& arguments) {
        std::uint64_t hash = op::gpu::hash(group.front().front().block_size);
        for (const auto& chain : group) {
            hash = op::gpu::hash(chain.size(), hash);
            for (const auto& member : chain)
                hash = op::gpu::hash(member.key, hash);
        }
        // Which parameters share a pointer shapes the kernel.
        return hash_bytes(arguments.buffer_index.data(),
                          arguments.buffer_index.size() * sizeof(int), hash);
    }

    static std::string group_name(const Group& group) {
        std::string name;
        for (const auto& chain : group) {
            name += name.empty() ? "" : " | ";
            for (std::size_t k = 0; k < chain.size(); ++k)
                name += (k == 0 ? "" : "+") + loop_of(chain[k]).fusion_name();
        }
        return name;
    }

    // The fused kernel's source: each member's constants and declarations
    // in a namespace of its own, so two members never collide, then the
    // wrapper.  Its pointer parameters carry no __restrict, as members may
    // pass one dat twice; each op2_body keeps its own.  Chain c takes
    // blocks_c blocks after those of the chains before it.
    static std::string source(Group& group, const Arguments& arguments) {
        std::string text =
            JitService::preamble(loop_of(group.front().front()).fusion_headers());
        std::size_t k = 0;
        for_members(group, [&](FusionMember& member) {
            text += "namespace op2_f" + std::to_string(k++) + " {\n";
            text += loop_of(member).fusion_params(member);
            text += member.strategy->source().decls;
            text += "}\n";
        });

        std::string signature, scalars;
        for (std::size_t b = 0; b < arguments.buffers.size(); ++b)
            signature += "    void *b" + std::to_string(b) + ",\n";
        std::string body;
        std::size_t buffer = 0, value = 0, scalar = 0;
        k = 0;
        for (std::size_t c = 0; c < group.size(); ++c) {
            auto chain = std::to_string(c);
            body += "    if (block < blocks" + chain + ") {\n"
                    "        const int thread_id = threadIdx.x + block * blockDim.x;\n"
                    "        for (int n = thread_id + start" + chain + "; n < end" + chain +
                    "; n += blockDim.x * blocks" + chain + ") {\n";
            for (const auto& member : group[c]) {
                const auto& fusion = *member.strategy->source().fusion;
                body += "            op2_f" + std::to_string(k++) + "::" + fusion.body + "(";
                for (const auto& param : fusion.params) {
                    if (param.kind == FusionParam::buffer ||
                        param.kind == FusionParam::global) {
                        body += "(" + std::string{param.type} + ")b" +
                                std::to_string(arguments.buffer_index[buffer++]) + ", ";
                    } else if (param.kind == FusionParam::value) {
                        std::string name = "v" + std::to_string(value++);
                        signature += "    " + std::string{param.type} + " " + name + ",\n";
                        body += name + ", ";
                    } else if (param.kind == FusionParam::scalar) {
                        std::string name = "s" + std::to_string(scalar++);
                        scalars += "    " + std::string{param.type} + " " + name + ",\n";
                        body += name + ", ";
                    }
                }
                body += "thread_id, n);\n";
            }
            body += "        }\n        return;\n    }\n    block -= blocks" + chain + ";\n";
        }
        std::string ranges;
        for (std::size_t c = 0; c < group.size(); ++c) {
            auto chain = std::to_string(c);
            ranges += "    const int start" + chain + ",\n    const int end" + chain +
                      ",\n    const int blocks" + chain +
                      (c + 1 < group.size() ? ",\n" : "\n");
        }

        text += "\nextern \"C\" __global__\n";
#ifdef OP2_HIP
        text += "__launch_bounds__(" +
                std::to_string(group.front().front().block_size) + ")\n";
#endif
        text += "void op2_fused(\n" + signature + scalars + ranges +
                ") {\n    int block = blockIdx.x;\n" + body + "}\n";
        return text;
    }

    // The group's kernel once compiled; starts its compile as the JIT
    // policy says.
    JitKernel *kernel(std::uint64_t hash, Group& group, const Arguments& arguments) {
        auto& entry = m_entries[hash];
        if (entry.rejected)
            return nullptr;
        if (entry.name.empty())
            entry.name = group_name(group);

        auto acquired = JitService::acquire_compiled(
            m_kernels, loop_of(group.front().front()).fusion_headers(), "op2_fused", hash,
            op::loop::Urgency::later, false,
            [&] {
                auto text = source(group, arguments);
                dump(entry.name, hash, text);
                return text;
            },
            entry.name);
        auto *compiled = acquired.kernel;
        if (compiled == nullptr || entry.checked)
            return compiled;

        // A fused kernel holding fewer than half the blocks per
        // multiprocessor its members do is rejected without timing.
        entry.checked = true;
        entry.registers = compiled->registers();
        int block_size = group.front().front().block_size;
        int member_occupancy = INT32_MAX;
        for_members(group, [&](const FusionMember& member) {
            if (member.registers > 0)
                member_occupancy = std::min(
                    member_occupancy, config().occupancy(member.registers, block_size));
        });
        int fused_occupancy = compiled->occupancy(block_size);
        entry.rejected =
            member_occupancy != INT32_MAX && 2 * fused_occupancy < member_occupancy;
        if (OP_diags > 2)
            op_printf("fusion: %s uses %d registers, %d blocks per SM (members %d)%s\n",
                      entry.name.c_str(), entry.registers, fused_occupancy,
                      member_occupancy == INT32_MAX ? 0 : member_occupancy,
                      entry.rejected ? ", rejected" : "");
        return entry.rejected ? nullptr : compiled;
    }

    // Write a fused kernel's source under OP_FUSION_DUMP.
    static void dump(const std::string& name, std::uint64_t hash, const std::string& text) {
        if (config().fusion_dump.empty())
            return;
        char file[32];
        std::snprintf(file, sizeof(file), "/op2_fused_%016lx.cu", hash);
        if (FILE *out = std::fopen((config().fusion_dump + file).c_str(), "w")) {
            std::fprintf(out, "// %s\n%s", name.c_str(), text.c_str());
            std::fclose(out);
        }
    }

    static void run_apart(Group& group) {
        for_members(group,
                    [](FusionMember& member) { member.call->loop->execute(*member.call); });
    }

    // The GPU time of run, in microseconds.
    template<typename Run>
    float timed(Run&& run) {
        for (auto& event : m_events)
            if (event == nullptr)
                CUDA_SAFE_CALL(gpuEventCreate(&event));
        CUDA_SAFE_CALL(gpuEventRecord(m_events[0], 0));
        run();
        CUDA_SAFE_CALL(gpuEventRecord(m_events[1], 0));
        CUDA_SAFE_CALL(gpuEventSynchronize(m_events[1]));
        float ms = 0.0f;
        CUDA_SAFE_CALL(gpuEventElapsedTime(&ms, m_events[0], m_events[1]));
        return ms * 1000.0f;
    }

    static float median(std::vector<float> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    void run_fused(Group& group, JitKernel& compiled, const std::string& name) {
        op_profile_enter_kernel(name.c_str(), "c_CUDA", "Fused");
        op_profile_enter("Computation");
        for_members(group, [](FusionMember& member) { loop_of(member).fused_begin(member); });
        // After the exchanges and the globals' preparation, which place the
        // globals.
        auto arguments = bind(group);

        std::vector<void *> argv;
        for (auto& buffer : arguments.buffers)
            argv.push_back(&buffer);
        for (auto& value : arguments.values)
            argv.push_back(&value);
        for (auto& scalar : arguments.scalars)
            argv.push_back(&scalar);
        // Each chain's range and blocks: the fewest any member allows.
        std::vector<int> ranges;
        ranges.reserve(3 * group.size());
        int blocks = 0;
        for (const auto& chain : group) {
            const auto& first = chain.front();
            int chain_blocks =
                (first.end - first.begin + first.block_size - 1) / first.block_size;
            for (const auto& member : chain)
                chain_blocks = std::min({chain_blocks, member.block_limit, member.max_blocks});
            ranges.insert(ranges.end(), {first.begin, first.end, chain_blocks});
            blocks += chain_blocks;
        }
        for (auto& range : ranges)
            argv.push_back(&range);
        if (blocks > 0)
            compiled.launch(blocks, group.front().front().block_size, argv.data(), 0);

        for_members(group, [](FusionMember& member) { loop_of(member).fused_end(member); });
        op_profile_exit();
        op_profile_exit();
        ++m_fused_launches;
        m_fused_calls += members(group);
    }

    void launch(Group& group) {
        if (members(group) == 1) {
            run_apart(group);
            return;
        }
        auto arguments = bind(group);
        auto hash = group_hash(group, arguments);
        auto *compiled = kernel(hash, group, arguments);
        auto& entry = m_entries[hash];
        if (compiled == nullptr) {
            run_apart(group);
            return;
        }
        if (!config().fusion_tune || entry.decided) {
            run_fused(group, *compiled, entry.name);
            return;
        }

        // Alternate timed runs, the first of each kind a warm-up.
        bool fused = entry.fused_times.size() <= entry.apart_times.size();
        float time = fused ? timed([&] { run_fused(group, *compiled, entry.name); })
                           : timed([&] { run_apart(group); });
        (fused ? entry.fused_times : entry.apart_times).push_back(time);
        if (entry.apart_times.size() < tune_samples + 1)
            return;
        auto samples = [](const std::vector<float>& times) {
            return std::vector<float>(times.begin() + 1, times.end());
        };
        float fused_time = median(samples(entry.fused_times));
        float apart_time = median(samples(entry.apart_times));
        entry.decided = true;
        entry.rejected = fused_time >= apart_time;
        if (OP_diags > 2)
            op_printf("fusion: %s %.1f us fused, %.1f us apart: %s\n", entry.name.c_str(),
                      fused_time, apart_time, entry.rejected ? "apart" : "fused");
    }

public:
    Fusion() { op::loop::register_release(this, &Fusion::release_callback); }
    ~Fusion() { op::loop::unregister_release(this); }
    Fusion(const Fusion&) = delete;

    // Run the queued calls in order, as fused kernels where they may share
    // one.
    void run(std::vector<PendingCall>& calls) {
        Group group;
        auto close = [&] {
            if (!group.empty())
                launch(group);
            group.clear();
        };
        for (auto& call : calls) {
            FusionMember member;
            bool eligible = config().fusion && call.loop->fusion_member(call, member) &&
                            fusable(member);
            if (!eligible) {
                close();
                call.loop->execute(call);
                continue;
            }
            if (!place(group, member)) {
                close();
                group.push_back({member});
            }
        }
        close();
    }
};

inline Fusion& fusion() {
    static Fusion instance;
    return instance;
}

} // namespace op::gpu
