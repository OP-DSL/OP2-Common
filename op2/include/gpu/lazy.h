#pragma once

// Lazy execution for the CUDA and HIP backend.  A call is captured into one
// process-wide queue and runs once the host needs its results: when a call
// writes a global back to the host, and wherever op_lazy_flush is called
// (fetches, uploads, timers, op_exit).

#include <gpu/config.h>
#include <loop/strategy.h>
#include <op_lib_core.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace op::gpu {

struct PendingCall;
struct FusionMember;
struct JitHeaders;

// What the queue runs a captured call through, and the operations a fused
// launch is made of (gpu/fusion.h).
class QueuedLoop {
public:
    virtual void execute(PendingCall& call) = 0;

    // Fill member from the call when the loop's preferred strategy can take
    // part in a fused kernel, restoring the call's JIT parameter values;
    // else say why not.
    virtual const char *fusion_member(PendingCall& call, FusionMember& member) = 0;
    // The member's JIT parameters as a compile's constants.
    virtual std::string fusion_params(const FusionMember& member) = 0;
    virtual const JitHeaders& fusion_headers() const = 0;
    virtual const std::string& fusion_name() const = 0;
    // Around a fused launch, after its halo exchange: the device globals'
    // preparation; their processing, once the owned elements have run; and
    // their reduction across the ranks and the dirty bits.
    virtual void fused_begin(FusionMember& member) = 0;
    virtual void fused_globals(FusionMember& member) = 0;
    virtual void fused_end(FusionMember& member) = 0;

protected:
    ~QueuedLoop() = default;
};

// Why the queue runs: for every rank at the same call (a global returned to
// the host), at op_exit, or for this rank alone (a fetch, a full queue after
// a local flush, ...).
enum class Flush { local, global, exit };

// One captured call: the set, a copy of the arguments, the values of its
// READ globals, and the loop's JIT parameter values.
struct PendingCall {
    QueuedLoop *loop = nullptr;
    op_set set = nullptr;
    std::vector<op_arg> args;
    std::vector<char> globals;
    // Each copied global's argument and offset in globals.
    std::vector<std::pair<int, std::size_t>> global_offsets;
    std::vector<std::byte> params;
    // The host reads a global back, or a READ global was too large to copy:
    // the queue runs with this call last.
    bool flush = false;

    // Point the copied globals' arguments at the copies.
    void bind_globals() {
        for (auto [index, offset] : global_offsets)
            args[static_cast<std::size_t>(index)].data = globals.data() + offset;
    }
};

// Whether an argument hands a value back to the host when the call ends.
inline bool returns_to_host(const op_arg& arg) {
    if (arg.opt == 0)
        return false;
    if (arg.argtype == OP_ARG_INFO)
        return true;
    return arg.argtype == OP_ARG_GBL && arg.acc != OP_READ && arg.acc != OP_WORK;
}

class Queue {
private:
    std::vector<PendingCall> m_calls;
    bool m_running = false;
    // Runs a batch of calls in order: the fusion engine.
    void (*m_runner)(std::vector<PendingCall>& calls, Flush flush) = nullptr;
    // For OP_diags: runs by reason, and calls run.
    std::map<std::string, std::size_t> m_runs;
    std::size_t m_executed = 0;

    static void release_callback(void *owner) {
        static_cast<Queue *>(owner)->report();
    }

    void report() {
        if (OP_diags > 1 && !m_runs.empty()) {
            std::size_t runs = 0;
            for (const auto& [reason, count] : m_runs)
                runs += count;
            op_printf("lazy: %zu calls in %zu runs (", m_executed, runs);
            const char *separator = "";
            for (const auto& [reason, count] : m_runs) {
                op_printf("%s%s %zu", separator, reason.c_str(), count);
                separator = ", ";
            }
            op_printf(")\n");
        }
        m_runs.clear();
        m_executed = 0;
    }

public:
    Queue() { op::loop::register_release(this, &Queue::release_callback); }
    ~Queue() { op::loop::unregister_release(this); }
    Queue(const Queue&) = delete;

    void set_runner(void (*runner)(std::vector<PendingCall>& calls, Flush flush)) {
        m_runner = runner;
    }

    // A call of loop over set, with its arguments and READ globals copied.
    static PendingCall capture(QueuedLoop& loop, op_set set, op_arg *args,
                               int nargs) {
        PendingCall call;
        call.loop = &loop;
        call.set = set;
        call.args.assign(args, args + nargs);
        for (int i = 0; i < nargs; ++i) {
            const auto& arg = args[i];
            call.flush |= returns_to_host(arg);
            if (arg.opt == 0 || arg.argtype != OP_ARG_GBL || arg.acc != OP_READ)
                continue;
            if (arg.size > config().lazy_copy_bytes) {
                call.flush = true;
                continue;
            }
            std::size_t offset = call.globals.size();
            call.globals.insert(call.globals.end(), arg.data, arg.data + arg.size);
            call.global_offsets.emplace_back(i, offset);
        }
        call.bind_globals();
        return call;
    }

    void push(PendingCall call) {
        bool flush = call.flush;
        m_calls.push_back(std::move(call));
        if (flush)
            run("global", Flush::global);
        else if (m_calls.size() >= static_cast<std::size_t>(config().lazy_max_calls))
            run("queue full", Flush::local);
    }

    // Run every queued call, in order.  Nothing runs it again meanwhile:
    // the uploads and downloads a call makes flush too.  op_exit reaches the
    // runner even with nothing queued.
    void run(const char *reason, Flush flush) {
        if (m_running)
            return;
        if (m_calls.empty()) {
            if (flush == Flush::exit && m_runner != nullptr)
                m_runner(m_calls, flush);
            return;
        }
        m_running = true;
        if (OP_diags > 2)
            op_printf("lazy: running %zu calls (%s)\n", m_calls.size(), reason);
        ++m_runs[reason];
        m_executed += m_calls.size();
        if (m_runner != nullptr)
            m_runner(m_calls, flush);
        else
            for (auto& call : m_calls)
                call.loop->execute(call);
        m_calls.clear();
        m_running = false;
    }
};

inline Queue& queue() {
    static Queue instance;
    return instance;
}

inline void flush_queue(const char *reason) {
    queue().run(reason, std::strcmp(reason, "exit") == 0 ? Flush::exit : Flush::local);
}

} // namespace op::gpu
