#pragma once

// Every loop the run made, which the reports read; the run's context owns
// it (op2/rt/context.h).  Defined in src/rt/loop/op_loop_report.cpp.

#include <op2/rt/abi/descriptor.h>

#include <memory>
#include <vector>

namespace op::rt {
struct Context;
}

namespace op::rt::loop {

class Loop;

class Registry {
private:
    // A loop and the descriptor that points at it.
    struct Registered {
        std::unique_ptr<Loop> loop;
        abi::Loop *descriptor;
    };

    std::vector<Registered> m_loops;
    // Whether write_reports printed the summaries.
    bool m_printed = false;

public:
    Registry();
    Registry(const Registry&) = delete;
    // Destroy every loop, which frees its device state, and clear its
    // descriptor, so a later call makes a new one.
    ~Registry();

    void add(std::unique_ptr<Loop> loop, abi::Loop& descriptor);

    // Write the tuner's and the JIT's reports under context's options,
    // OP_AUTOTUNE_REPORT and OP_JIT_REPORT, suffixed with the rank when there
    // is more than one, and on rank 0 print their summaries of this rank's
    // loops; at the end only if no earlier call printed them.  A later call
    // writes the files again, so applications that never call op_exit still
    // get them.
    void write_reports(const Context& context, int rank, int ranks, bool end);
};

} // namespace op::rt::loop
