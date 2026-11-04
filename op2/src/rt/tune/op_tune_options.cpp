#include <op2/rt/tune/options.h>

#include <op2/env.h>

namespace op::rt::tune {

AutotuneOptions read_options() {
    AutotuneOptions options;
    constexpr env::Choice modes[] = {{"off", 0}, {"0", 0},      {"no", 0},
                                     {"false", 0}, {"report", 1}, {"select", 2}};
    int mode = 0;
    env::parse_choice("OP_AUTOTUNE", modes, mode);
    options.report = mode > 0;
    options.select = mode == 2;
    env::parse_text("OP_AUTOTUNE_REPORT", options.report_file);
    env::parse_number("OP_AUTOTUNE_MARGIN", options.margin);
    env::parse_number("OP_AUTOTUNE_WARMUP", options.warmup);
    env::parse_number("OP_AUTOTUNE_SAMPLES", options.samples, 1);
    env::parse_number("OP_AUTOTUNE_MIN_CALLS", options.min_calls);
    return options;
}

} // namespace op::rt::tune
