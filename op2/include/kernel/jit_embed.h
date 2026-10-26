#pragma once

// Embed, as text, the headers a JIT compile of this translation unit's loops
// needs.  Define OP_KERNEL_JIT_TAG to a token unique to the unit first; the
// JIT compile reads the OP_*_DATA macros.

#ifndef OP_KERNEL_JIT_TAG
#error "define OP_KERNEL_JIT_TAG before including kernel/jit_embed.h"
#endif

#ifndef INCBIN_STYLE
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#endif
#include <extern/incbin.h>

#define OP_KERNEL_JIT_CAT2(a, b) a##b
#define OP_KERNEL_JIT_CAT(a, b) OP_KERNEL_JIT_CAT2(a, b)

#define OP_KERNEL_PRELUDE OP_KERNEL_JIT_CAT(op_kernel_prelude_, OP_KERNEL_JIT_TAG)
#define OP_KERNEL_PRELUDE_DATA OP_KERNEL_JIT_CAT(OP_KERNEL_PRELUDE, _data)
INCTXT(OP_KERNEL_PRELUDE, "kernel/prelude.h");

#define OP_STAGING_LAYOUT OP_KERNEL_JIT_CAT(op_staging_layout_, OP_KERNEL_JIT_TAG)
#define OP_STAGING_LAYOUT_DATA OP_KERNEL_JIT_CAT(OP_STAGING_LAYOUT, _data)
INCTXT(OP_STAGING_LAYOUT, "plan/staging_layout.h");

#define OP_STRATEGY_KERNEL OP_KERNEL_JIT_CAT(op_strategy_kernel_, OP_KERNEL_JIT_TAG)
#define OP_STRATEGY_KERNEL_DATA OP_KERNEL_JIT_CAT(OP_STRATEGY_KERNEL, _data)
INCTXT(OP_STRATEGY_KERNEL, "kernel/strategy.h");

// The embedded headers as the backend's JIT compiler takes them; the
// application's constants header when the unit embedded one as
// OP_KERNEL_PARAMS.
#ifdef OP_KERNEL_PARAMS
#define OP_KERNEL_JIT_HEADERS                                                     \
    ::op::gpu::JitHeaders {                                                    \
        {OP_KERNEL_PRELUDE_DATA, OP_STAGING_LAYOUT_DATA, OP_STRATEGY_KERNEL_DATA, \
         OP_KERNEL_PARAMS_DATA},                                                  \
        {"kernel/prelude.h", "plan/staging_layout.h", "kernel/strategy.h",     \
         "op_f2c_params.h"},                                                   \
        true                                                                   \
    }
#else
#define OP_KERNEL_JIT_HEADERS                                                     \
    ::op::gpu::JitHeaders {                                                    \
        {OP_KERNEL_PRELUDE_DATA, OP_STAGING_LAYOUT_DATA, OP_STRATEGY_KERNEL_DATA}, \
        {"kernel/prelude.h", "plan/staging_layout.h", "kernel/strategy.h"},    \
        false                                                                  \
    }
#endif
