#include <op2/rt/jit/options.h>

#include <op2/env.h>
#include <op_lib_core.h>

namespace op::rt::jit {

Options read_options() {
    Options options;
    env::retired("OP_JIT_ENABLE", "OP_JIT_LEVEL=0 turns the JIT off");
    env::parse_number("OP_JIT_LEVEL", options.level, 0, 2);
    if (!options.enabled())
        op_printf("Disabling JIT compilation\n");
    env::parse_bool("OP_JIT_DEBUG", options.debug);
    if (options.debug)
        op_printf("Enabling JIT debug\n");
    env::parse_bool("OP_JIT_SEQ_COMPILE", options.seq_compile);
    env::parse_bool("OP_JIT_FORCE", options.force);
    if (options.force)
        op_printf("Forcing JIT compilation of every kernel\n");
    env::parse_number("OP_JIT_MIN_CALLS", options.min_calls);
    env::parse_number("OP_JIT_MAX_THREADS", options.max_threads, 1);
    env::parse_number("OP_JIT_VARYING", options.varying);
    env::parse_number("OP_JIT_MAX_GBL_BYTES", options.max_gbl_bytes);
    env::parse_number("OP_JIT_MAX_CONST_BYTES", options.max_const_bytes);
    env::parse_list("OP_JIT_RUNTIME_PARAMS", options.runtime_params);
    env::parse_list("OP_JIT_CONSTANT_PARAMS", options.constant_params);
    env::parse_text("OP_JIT_REPORT", options.report_file);
    return options;
}

} // namespace op::rt::jit
