#pragma once

// Runtime kernel fusion for the CUDA and HIP backend.  When the lazy queue
// runs, its calls are grouped into fused kernels.  A chain is calls over one
// set whose dependencies are element-wise: for each element the kernel runs
// each call's op2_body in queue order.  Independent chains share a kernel in
// disjoint block ranges.  Each call's strategy supplies its declarations and
// its element function (FusionSource); only the wrapper is written here.  A
// fused kernel exchanges every member's halo at once and launches as the
// strategies do: the elements that need no halo while it arrives, then the
// rest.

#include <gpu/agreement.h>
#include <gpu/jit.h>
#include <gpu/lazy.h>
#include <gpu/runs.h>
#include <gpu/strategy.h>
#include <loop/fusion.h>
#include <op_lib_core.h>
#include <op_profile.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace op::gpu {

// A queued call as a member of a fused kernel, from its loop's preferred
// strategy.
struct FusionMember {
    PendingCall *call = nullptr;
    StrategyBase *strategy = nullptr;
    // The elements, those of them that need no halo, the owned ones, and
    // the arrays' stride; whether the elements include the exec halo, which
    // unlike their counts is the same on every rank.
    int begin = 0;
    int end = 0;
    int core = 0;
    int owned = 0;
    int set_stride = 0;
    bool exec = false;
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


class Fusion final : public Runner {
private:
    using Chain = std::vector<FusionMember>;
    using Group = std::vector<Chain>;

    struct Entry {
        std::string name;
        bool checked = false;
        // Its kernel holds too few blocks per multiprocessor here.
        bool rejected = false;
        int registers = 0;
        // Timed runs, fused and apart, alternating from fused, once every
        // rank has the kernel; this rank's medians go to the agreement once
        // each kind has its warm-up and samples.
        std::vector<float> fused_times;
        std::vector<float> apart_times;
        bool timed = false;
        bool reported = false;
    };

    static constexpr std::size_t tune_samples = 3;

    JitKernels m_kernels;
    gpuEvent_t m_events[2] = {};
    std::unordered_map<std::uint64_t, Entry> m_entries;
    Agreement m_agreement;
    std::size_t m_fused_launches = 0;
    std::size_t m_fused_calls = 0;
    // For OP_diags: the run layouts printed, and why groups closed, by kind.
    std::unordered_set<std::uint64_t> m_reported;
    std::map<std::string, std::size_t> m_closes;
    // For OP_diags > 1: the runs' shapes, and the current run's launches.
    RunShapes m_shapes;
    std::size_t m_run_launches = 0;

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
        if (OP_diags > 1) {
            for (const auto& [kind, count] : m_closes)
                op_printf("fusion: %zu closed by %s\n", count, kind.c_str());
            m_shapes.report();
        }
        m_fused_launches = 0;
        m_fused_calls = 0;
        m_closes.clear();
        m_reported.clear();
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

    // Whether member and other run the same elements.  Grouping looks only
    // at what every rank sees alike, so the ranks fuse alike, though their
    // exchanges would match regardless (op_mpi_halo_exchanges_batch).
    static bool same_range(const FusionMember& member, const FusionMember& other) {
        return member.call->set == other.call->set && member.exec == other.exec;
    }

    // Whether member may run after every call of chain, element by element.
    static bool joins(const Chain& chain, const FusionMember& member,
                      const std::vector<op::loop::fusion::Use>& member_uses) {
        if (!same_range(member, chain.front()))
            return false;
        for (const auto& other : chain) {
            auto other_uses = uses(other);
            if (op::loop::fusion::conflict(other_uses, atomic(other), member_uses,
                                           atomic(member), true, member.exec)
                    .first != nullptr)
                return false;
        }
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

    enum class Placed { nowhere, chain, apart };

    static bool device_globals_clash(Group& group, const FusionMember& member) {
        int device_globals = member.device_globals;
        for_members(group, [&](const FusionMember& other) {
            device_globals += other.device_globals;
        });
        return device_globals > 1;
    }

    // Place member in the group: in the first chain it may follow, being
    // independent of the others, else in a chain of its own, being
    // independent of all.
    static Placed place(Group& group, const FusionMember& member) {
        if (group.empty()) {
            group.push_back({member});
            return Placed::apart;
        }
        if (member.block_size != group.front().front().block_size ||
            device_globals_clash(group, member))
            return Placed::nowhere;

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
                return Placed::chain;
            }
        }
        if (std::all_of(apart.begin(), apart.end(), [](bool b) { return b; })) {
            group.push_back({member});
            return Placed::apart;
        }
        return Placed::nowhere;
    }

    // Why member fits nowhere in group, for reports: its first conflict
    // with the latest member it has one with, and that conflict's kind.
    static std::pair<std::string, std::string> explain(Group& group,
                                                       const FusionMember& member) {
        if (member.block_size != group.front().front().block_size)
            return {"block sizes differ", "block sizes"};
        if (device_globals_clash(group, member))
            return {"both take device globals", "device globals"};
        auto member_uses = uses(member);
        for (auto chain = group.rbegin(); chain != group.rend(); ++chain)
            for (auto other = chain->rbegin(); other != chain->rend(); ++other) {
                bool same_set = other->call->set == member.call->set;
                bool in_chain = same_range(member, *other);
                auto other_uses = uses(*other);
                auto found = op::loop::fusion::conflict(other_uses, atomic(*other),
                                                        member_uses, atomic(member), in_chain,
                                                        member.exec);
                if (found.first == nullptr)
                    continue;
                bool halo = in_chain && op::loop::fusion::element_wise(
                                         *found.first, atomic(*other), *found.second,
                                         atomic(member));
                std::string kind = op::loop::fusion::access(*found.first) + " -> " +
                                   op::loop::fusion::access(*found.second) +
                                   (halo       ? ", same set, read in the exec halo"
                                    : in_chain ? ", same set"
                                    : same_set ? ", same set, other range"
                                               : ", other sets");
                std::string detail = loop_of(*other).fusion_name() + " -> " +
                                     loop_of(member).fusion_name() + " on " +
                                     found.first->dat->name + " (" +
                                     other->call->set->name + " -> " +
                                     member.call->set->name + "): " + kind;
                return {detail, kind};
            }
        return {"it would join two chains", "two chains"};
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

    void run_apart(Group& group) {
        for_members(group,
                    [](FusionMember& member) { member.call->loop->execute(*member.call); });
        m_run_launches += members(group);
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
        op_mpi_drain();
        CUDA_SAFE_CALL(gpuEventSynchronize(m_events[1]));
        float ms = 0.0f;
        CUDA_SAFE_CALL(gpuEventElapsedTime(&ms, m_events[0], m_events[1]));
        return ms * 1000.0f;
    }

    static float median(std::vector<float> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    // One halo exchange for every member.
    static void exchange(Group& group) {
        std::vector<op_set> sets;
        std::vector<int> nargs;
        std::vector<op_arg *> args;
        for_members(group, [&](FusionMember& member) {
            member.call->bind_globals();
            sets.push_back(member.call->set);
            nargs.push_back(static_cast<int>(member.call->args.size()));
            args.push_back(member.call->args.data());
        });
        op_mpi_halo_exchanges_batch(static_cast<int>(sets.size()), sets.data(), nargs.data(),
                                    args.data(), 2);
    }

    // A chain's elements in phase 0, before the halo wait: those no member
    // needs the halo for, none without OP_FUSION_OVERLAP; 1: the rest of the
    // owned ones, or of all when no member writes a global; 2: the exec halo,
    // after the globals' processing.
    static std::pair<int, int> phase_range(const Chain& chain, int phase) {
        const auto& first = chain.front();
        int core = config().fusion_overlap ? first.end : first.begin;
        bool globals = false;
        for (const auto& member : chain) {
            core = std::min(core, member.core);
            globals |= member.global_output;
        }
        int owned = globals ? first.owned : first.end;
        switch (phase) {
        case 0:
            return {first.begin, core};
        case 1:
            return {core, std::max(core, owned)};
        default:
            return {std::max(core, owned), first.end};
        }
    }

    void run_fused(Group& group, JitKernel& compiled, const std::string& name) {
        op_profile_enter_kernel(name.c_str(), "c_CUDA", "Fused");
        op_profile_enter("MPI Exchanges");
        exchange(group);
        op_profile_next("Prepare GBLs");
        for_members(group, [](FusionMember& member) { loop_of(member).fused_begin(member); });
        // After the globals' preparation, which places them.
        auto arguments = bind(group);

        std::vector<void *> argv;
        for (auto& buffer : arguments.buffers)
            argv.push_back(&buffer);
        for (auto& value : arguments.values)
            argv.push_back(&value);
        for (auto& scalar : arguments.scalars)
            argv.push_back(&scalar);
        std::vector<int> ranges(3 * group.size());
        for (auto& range : ranges)
            argv.push_back(&range);

        op_profile_next("Computation");
        for (int phase = 0; phase < 3; ++phase) {
            if (phase == 1) {
                op_profile_next("MPI Wait");
                op_mpi_wait_all(0, nullptr);
                op_profile_next("Computation");
            }
            // Each chain's range and blocks: the fewest any member allows.
            int blocks = 0;
            for (std::size_t c = 0; c < group.size(); ++c) {
                const auto& chain = group[c];
                auto [begin, end] = phase_range(chain, phase);
                int chain_blocks = (end - begin + chain.front().block_size - 1) /
                                   chain.front().block_size;
                for (const auto& member : chain)
                    chain_blocks =
                        std::min({chain_blocks, member.block_limit, member.max_blocks});
                ranges[3 * c] = begin;
                ranges[3 * c + 1] = end;
                ranges[3 * c + 2] = std::max(chain_blocks, 0);
                blocks += ranges[3 * c + 2];
            }
            if (blocks > 0)
                compiled.launch(blocks, group.front().front().block_size, argv.data(), 0);
            if (phase == 1)
                for_members(group,
                            [](FusionMember& member) { loop_of(member).fused_globals(member); });
        }

        for_members(group, [](FusionMember& member) { loop_of(member).fused_end(member); });
        op_profile_exit();
        op_profile_exit();
        ++m_fused_launches;
        ++m_run_launches;
        m_fused_calls += members(group);
    }

    // Run the group fused or apart as every rank agrees (gpu/agreement.h):
    // apart until all have its kernel, then timed both ways where tuning,
    // then as the slowest rank's times say.
    void launch(Group& group) {
        if (members(group) == 1) {
            run_apart(group);
            return;
        }
        auto arguments = bind(group);
        auto hash = group_hash(group, arguments);
        auto& entry = m_entries[hash];
        m_agreement.seen(hash);
        auto *compiled = kernel(hash, group, arguments);
        if (compiled != nullptr || entry.rejected)
            m_agreement.ready(hash, entry.rejected);
        const auto& agreed = m_agreement.agreed(hash);
        if (agreed.not_ready != 0.0 || compiled == nullptr) {
            run_apart(group);
            return;
        }
        if (!config().fusion_tune) {
            run_fused(group, *compiled, entry.name);
            return;
        }
        if (agreed.fused >= 0.0 && agreed.apart >= 0.0) {
            bool fused = agreed.fused < agreed.apart;
            if (OP_diags > 2 && !entry.reported)
                op_printf("fusion: %s %.1f us fused, %.1f us apart on the slowest rank: %s\n",
                          entry.name.c_str(), agreed.fused, agreed.apart,
                          fused ? "fused" : "apart");
            entry.reported = true;
            if (fused)
                run_fused(group, *compiled, entry.name);
            else
                run_apart(group);
            return;
        }
        if (entry.timed) {
            run_apart(group);
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
        entry.timed = true;
        m_agreement.timed(hash, median(samples(entry.fused_times)),
                          median(samples(entry.apart_times)));
    }

    // The kernel being gathered, and for OP_diags the run so far: its calls'
    // shapes, launches, layout (groups apart by " / ", chains by " | ",
    // members of a chain by "+") and why groups closed.
    Group m_group;
    std::vector<RunCall> m_run;
    std::vector<const std::string *> m_run_names;
    std::string m_layout, m_reasons;

    void close() {
        if (!m_group.empty())
            launch(m_group);
        m_group.clear();
        op_mpi_progress();
    }

    void note(const char *separator, const std::string& name) {
        if (OP_diags > 2)
            m_layout += (m_layout.empty() ? "" : separator) + name;
    }

    void closed(const std::string& name, const std::string& detail, const std::string& kind) {
        ++m_closes[kind];
        if (OP_diags > 2)
            m_reasons += "  " + name + ": " + detail + "\n";
    }

public:
    Fusion() {
        op::loop::register_release(this, &Fusion::release_callback);
    }
    ~Fusion() { op::loop::unregister_release(this); }
    Fusion(const Fusion&) = delete;

    // The next queued call: it joins the kernel being gathered, or that
    // kernel launches and the call starts the next, or runs alone.
    void push(PendingCall& call) override {
        if (!config().fusion) {
            call.loop->execute(call);
            return;
        }
        FusionMember member;
        const char *why = call.loop->fusion_member(call, member);
        if (why == nullptr && !fusable(member))
            why = "registers";
        const auto& name = call.loop->fusion_name();
        if (OP_diags > 1) {
            auto& shape = m_run.emplace_back();
            shape.loop = std::hash<std::string>{}(name);
            shape.set = call.set;
            shape.args = call.args;
            shape.atomic = member.strategy != nullptr && atomic(member);
            shape.fusable = why == nullptr;
            for (const auto& arg : call.args)
                shape.dirty.push_back(arg.opt != 0 && arg.argtype == OP_ARG_DAT &&
                                      arg.dat->dirtybit == 1);
            m_run_names.push_back(&name);
        }
        if (why != nullptr) {
            if (OP_diags > 1)
                closed(name, std::string{"runs alone: "} + why,
                       std::string{"a call that runs alone: "} + why);
            note(" / ", "(" + name + ")");
            close();
            call.loop->execute(call);
            ++m_run_launches;
            return;
        }
        bool fresh = m_group.empty();
        auto placed = place(m_group, member);
        if (placed == Placed::nowhere) {
            if (OP_diags > 1) {
                auto [detail, kind] = explain(m_group, member);
                closed(name, detail, kind);
            }
            close();
            m_group.push_back({member});
            fresh = true;
        }
        note(fresh ? " / " : placed == Placed::chain ? "+" : " | ", name);
    }

    // The queue flushes: the last kernel launches.  At a global flush every
    // rank is here at the same call, so the agreement takes its round.
    void flush(Flush flush) override {
        if (flush == Flush::exit)
            m_agreement.finish();
        close();
        // What flushed the queue may block on the device next.
        op_mpi_drain();
        if (flush == Flush::global)
            m_agreement.round();
        if (OP_diags > 1 && !m_run.empty()) {
            m_shapes.observe(m_run, m_run_names, m_run_launches);
            if (OP_diags > 2 &&
                m_reported.insert(hash_bytes(m_layout.data(), m_layout.size(), 0)).second)
                op_printf("fusion: layout %s\n%s", m_layout.c_str(), m_reasons.c_str());
        }
        m_run.clear();
        m_run_names.clear();
        m_layout.clear();
        m_reasons.clear();
        m_run_launches = 0;
    }
};

inline Fusion& fusion() {
    static Fusion instance;
    return instance;
}

} // namespace op::gpu
