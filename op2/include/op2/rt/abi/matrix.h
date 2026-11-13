#pragma once

// How a matrix argument's blocks are found (op_arg_mat): the contract between
// a matrix's storage, which a solver library or OP2 holds, and the kernels.
// Compiled on the host and for the device, by NVRTC too, so it includes
// nothing.

namespace op::rt::abi::mat {

/* A matrix's values are in up to three regions: a library's local and
   non-local parts, say, and OP2's scratch for blocks of halo rows.  Its
   layout holds, per slot of its sparsity, an entry of four ints: entry (r, c)
   of the slot's d x d block is at offset + r * row_stride + c * col_stride in
   region aux & region_mask.  A diagonal block may move its diagonal entries
   by the rule aux >> rule_shift:
     diagonal_first     each row's diagonal entry leads it, the others
                        following in order (hypre's diagonal part);
     diagonal_separate  entry (r, r) at col_stride + r instead, the others
                        in their rows without it, column stride 1 (MCSR). */
constexpr int entry = 4;
constexpr int offset = 0, row_stride = 1, col_stride = 2, aux = 3;

constexpr int regions = 3;
constexpr int region_mask = 3, rule_shift = 2;

constexpr int plain = 0, diagonal_first = 1, diagonal_separate = 2;

/* How kernels find a slot's block, the least general form its layout fits,
   which a specialised kernel bakes in so the rest compiles out.  Where it is
   (form & where_mask):
     contiguous  block s at s * d * d of region 0, row major, its diagonal
                 entries where they are: the layout is not read;
     one_region  through the layout, every slot in region 0;
     any_region  through the layout, in its slot's region;
   with unit_columns every block's column stride is 1, so a row's entries
   are at constant offsets from it; and form >> form_rule_shift is the one
   rule its slots may have (plain when none has one). */
constexpr int contiguous = 0, one_region = 1, any_region = 2, where_mask = 3;
constexpr int unit_columns = 4;
constexpr int form_rule_shift = 3;

} // namespace op::rt::abi::mat

/* A matrix's storage as generated code binds it: its regions in the space
   each variant's loops use, its layout there and the form kernels find its
   blocks by.  Untranslated loops
   (op_seq.h) reach its blocks one after another in a buffer of the matrix's
   own, which untranslated(self, acc) returns before such a loop runs: the
   current values for OP_READ, zeros to add into for OP_INC. */
struct op_mat_storage {
    int dim;
    int form;
    void *region[op::rt::abi::mat::regions];
    void *region_d[op::rt::abi::mat::regions];
    const int *layout;
    const int *layout_d;
    char *(*untranslated)(op_mat_storage *self, int acc);
};
