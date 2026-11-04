#pragma once

// Embed, as text, the headers a JIT compile of this translation unit's loops
// includes.  Define OP_KERNEL_JIT_TAG to a token unique to the unit first,
// and OP_KERNEL_JIT_PARAMS to the path of an application's constants header
// to embed last, if any; OP_KERNEL_JIT_HEADERS is them as the backend's JIT
// compiler takes them.

#ifndef OP_KERNEL_JIT_TAG
#error "define OP_KERNEL_JIT_TAG before including op2/rt/kernel/gpu/jit_embed.h"
#endif

#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include <extern/incbin.h>

#define OP_KERNEL_JIT_CAT2(a, b) a##b
#define OP_KERNEL_JIT_CAT(a, b) OP_KERNEL_JIT_CAT2(a, b)

// The headers, in the order a specialised kernel's source includes them.
#define OP_KERNEL_JIT_FILES(X)                                                     \
    X(op_kernel_prelude_, "op2/rt/kernel/prelude.h")                               \
    X(op_abi_staging_, "op2/rt/abi/staging.h")                                     \
    X(op_abi_launch_, "op2/rt/abi/gpu/launch.h")                                   \
    X(op_kernel_inc_, "op2/rt/kernel/inc.h")                                       \
    X(op_kernel_staging_, "op2/rt/kernel/staging.h")                               \
    X(op_kernel_device_, "op2/rt/kernel/gpu/device.h")                             \
    X(op_kernel_params_, "op2/rt/kernel/gpu/params.h")                             \
    X(op_kernel_wavefront_, "op2/rt/kernel/gpu/wavefront.h")                       \
    X(op_kernel_range_, "op2/rt/kernel/gpu/range.h")                               \
    X(op_kernel_chunk_, "op2/rt/kernel/gpu/chunk.h")

#define OP_KERNEL_JIT_EMBED(name, path)                                            \
    INCTXT(OP_KERNEL_JIT_CAT(name, OP_KERNEL_JIT_TAG), path);
#define OP_KERNEL_JIT_DATA(name, path)                                             \
    OP_KERNEL_JIT_CAT(OP_KERNEL_JIT_CAT(name, OP_KERNEL_JIT_TAG), _data),
#define OP_KERNEL_JIT_NAME(name, path) path,

OP_KERNEL_JIT_FILES(OP_KERNEL_JIT_EMBED)
#ifdef OP_KERNEL_JIT_PARAMS
OP_KERNEL_JIT_EMBED(op_app_params_, OP_KERNEL_JIT_PARAMS)
#endif

[[maybe_unused]] static const char *const OP_KERNEL_JIT_CAT(op_kernel_jit_data_,
                                                            OP_KERNEL_JIT_TAG)[] = {
    OP_KERNEL_JIT_FILES(OP_KERNEL_JIT_DATA)
#ifdef OP_KERNEL_JIT_PARAMS
    OP_KERNEL_JIT_DATA(op_app_params_, OP_KERNEL_JIT_PARAMS)
#endif
};

[[maybe_unused]] static const char *const OP_KERNEL_JIT_CAT(op_kernel_jit_names_,
                                                            OP_KERNEL_JIT_TAG)[] = {
    OP_KERNEL_JIT_FILES(OP_KERNEL_JIT_NAME)
#ifdef OP_KERNEL_JIT_PARAMS
    OP_KERNEL_JIT_PARAMS,
#endif
};

#undef OP_KERNEL_JIT_FILES
#undef OP_KERNEL_JIT_EMBED
#undef OP_KERNEL_JIT_DATA
#undef OP_KERNEL_JIT_NAME

#define OP_KERNEL_JIT_HEADERS                                                      \
    ::op::rt::abi::gpu::JitHeaders {                                               \
        OP_KERNEL_JIT_CAT(op_kernel_jit_data_, OP_KERNEL_JIT_TAG),                 \
            OP_KERNEL_JIT_CAT(op_kernel_jit_names_, OP_KERNEL_JIT_TAG)             \
    }
