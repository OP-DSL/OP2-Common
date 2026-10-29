#pragma once

// Embed, as text, the headers a JIT compile of this translation unit's loops
// includes.  Define OP_KERNEL_JIT_TAG to a token unique to the unit first;
// OP_KERNEL_JIT_HEADERS is them as the backend's JIT compiler takes them.

#ifndef OP_KERNEL_JIT_TAG
#error "define OP_KERNEL_JIT_TAG before including op2/rt/kernel/gpu/jit_embed.h"
#endif

#ifndef INCBIN_STYLE
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#endif
#include <extern/incbin.h>

#define OP_KERNEL_JIT_CAT2(a, b) a##b
#define OP_KERNEL_JIT_CAT(a, b) OP_KERNEL_JIT_CAT2(a, b)

#define OP_KERNEL_JIT_EMBED(name, path)                                            \
    INCTXT(OP_KERNEL_JIT_CAT(name, OP_KERNEL_JIT_TAG), path)

OP_KERNEL_JIT_EMBED(op_kernel_prelude_, "op2/rt/kernel/prelude.h");
OP_KERNEL_JIT_EMBED(op_abi_staging_, "op2/rt/abi/staging.h");
OP_KERNEL_JIT_EMBED(op_kernel_inc_, "op2/rt/kernel/inc.h");
OP_KERNEL_JIT_EMBED(op_kernel_staging_, "op2/rt/kernel/staging.h");
OP_KERNEL_JIT_EMBED(op_kernel_wavefront_, "op2/rt/kernel/gpu/wavefront.h");

#define OP_KERNEL_JIT_DATA(name) OP_KERNEL_JIT_CAT(OP_KERNEL_JIT_CAT(name, OP_KERNEL_JIT_TAG), _data)

// The headers a specialised kernel's source includes, in order, as the
// backend's JIT compiler takes them: the application's constants header last
// when the unit embedded one as OP_KERNEL_PARAMS.
#define OP_KERNEL_JIT_KERNEL_DATA                                                  \
    OP_KERNEL_JIT_DATA(op_kernel_prelude_), OP_KERNEL_JIT_DATA(op_abi_staging_),   \
        OP_KERNEL_JIT_DATA(op_kernel_inc_), OP_KERNEL_JIT_DATA(op_kernel_staging_), \
        OP_KERNEL_JIT_DATA(op_kernel_wavefront_)
#define OP_KERNEL_JIT_KERNEL_NAMES                                                 \
    "op2/rt/kernel/prelude.h", "op2/rt/abi/staging.h", "op2/rt/kernel/inc.h",     \
        "op2/rt/kernel/staging.h", "op2/rt/kernel/gpu/wavefront.h"

#ifdef OP_KERNEL_PARAMS
#define OP_KERNEL_JIT_HEADERS                                                      \
    ::op::rt::gpu::JitHeaders {                                                    \
        {OP_KERNEL_JIT_KERNEL_DATA, OP_KERNEL_PARAMS_DATA},                        \
        {OP_KERNEL_JIT_KERNEL_NAMES, "op_f2c_params.h"}                            \
    }
#else
#define OP_KERNEL_JIT_HEADERS                                                      \
    ::op::rt::gpu::JitHeaders {                                                    \
        {OP_KERNEL_JIT_KERNEL_DATA}, {OP_KERNEL_JIT_KERNEL_NAMES}                  \
    }
#endif
