#pragma once

// Embed, as text, the headers a JIT compile of this translation unit's loops
// needs.  Define OP_F2C_JIT_TAG to a token unique to the unit first; the
// JIT compile reads the OP_*_DATA macros.

#ifndef OP_F2C_JIT_TAG
#error "define OP_F2C_JIT_TAG before including kernel/jit_embed.h"
#endif

#ifndef INCBIN_STYLE
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#endif
#include <extern/incbin.h>

#define OP_F2C_JIT_CAT2(a, b) a##b
#define OP_F2C_JIT_CAT(a, b) OP_F2C_JIT_CAT2(a, b)

#define OP_F2C_PRELUDE OP_F2C_JIT_CAT(op_f2c_prelude_, OP_F2C_JIT_TAG)
#define OP_F2C_PRELUDE_DATA OP_F2C_JIT_CAT(OP_F2C_PRELUDE, _data)
INCTXT(OP_F2C_PRELUDE, "kernel/prelude.h");

#define OP_CHUNK_LAYOUT OP_F2C_JIT_CAT(op_chunk_layout_, OP_F2C_JIT_TAG)
#define OP_CHUNK_LAYOUT_DATA OP_F2C_JIT_CAT(OP_CHUNK_LAYOUT, _data)
INCTXT(OP_CHUNK_LAYOUT, "loop/chunk_layout.h");

#define OP_STRATEGY_KERNEL OP_F2C_JIT_CAT(op_strategy_kernel_, OP_F2C_JIT_TAG)
#define OP_STRATEGY_KERNEL_DATA OP_F2C_JIT_CAT(OP_STRATEGY_KERNEL, _data)
INCTXT(OP_STRATEGY_KERNEL, "kernel/strategy.h");
