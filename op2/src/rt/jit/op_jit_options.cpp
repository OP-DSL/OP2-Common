#include <op2/rt/jit/options.h>

#include <op2/env.h>
#include <op_lib_core.h>

namespace op::rt::jit {

namespace {

Options run_options;

} // namespace

const Options& options() { return run_options; }

void init() {
    run_options = {};
    env::retired("OP_JIT_ENABLE", "OP_JIT_LEVEL=0 turns the JIT off");
    env::parse_number("OP_JIT_LEVEL", run_options.level, 0, 2);
    if (!run_options.enabled())
        op_printf("Disabling JIT compilation\n");
    env::parse_bool("OP_JIT_DEBUG", run_options.debug);
    if (run_options.debug)
        op_printf("Enabling JIT debug\n");
    env::parse_bool("OP_JIT_SEQ_COMPILE", run_options.seq_compile);
    env::parse_bool("OP_JIT_FORCE", run_options.force);
    if (run_options.force)
        op_printf("Forcing JIT compilation of every kernel\n");
    env::parse_number("OP_JIT_MIN_CALLS", run_options.min_calls);
    env::parse_number("OP_JIT_MAX_THREADS", run_options.max_threads, 1);
    env::parse_number("OP_JIT_VARYING", run_options.varying);
    env::parse_number("OP_JIT_MAX_GBL_BYTES", run_options.max_gbl_bytes);
    env::parse_number("OP_JIT_MAX_CONST_BYTES", run_options.max_const_bytes);
    env::parse_list("OP_JIT_RUNTIME_PARAMS", run_options.runtime_params);
    env::parse_list("OP_JIT_CONSTANT_PARAMS", run_options.constant_params);
    env::parse_text("OP_JIT_REPORT", run_options.report_file);
}

} // namespace op::rt::jit
