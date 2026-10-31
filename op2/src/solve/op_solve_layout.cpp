// Where each slot's block is in a solver library's storage format: the layout
// entries loops find blocks by (op2/rt/abi/matrix.h), and the format's index
// arrays for the library.  Host algorithms over a sparsity's block CSR.

#include <op2/solve/solve.h>

#include <algorithm>
#include <climits>

namespace op::solve {

namespace {

// A part's block CSR, and the slot of its first block.
struct Blocks {
    const std::vector<int>& ptr;
    const std::vector<int>& col;
    int first;
    bool diag;
};

Blocks blocks(op_mat A, Part part) {
    op_sparsity sp = A->sparsity;
    if (part == Part::diag)
        return {sp->diag_ptr, sp->diag_col, 0, true};
    return {sp->offd_ptr, sp->offd_col, sp->diag_blocks, false};
}

Entry entry(long offset, int rs, int cs, int region, int rule = mat::plain) {
    if (offset > INT_MAX)
        fail("matrix values past 2^31 in one region");
    return {(int)offset, rs, cs, region | rule << mat::rule_shift};
}

// Whether the sparsity's rows lead with their diagonal block.
bool diagonal_leads(op_mat A) {
    op_sparsity sp = A->sparsity;
    for (int I = 0; I < sp->rows->size; I++)
        if (sp->diag_ptr[I] == sp->diag_ptr[I + 1] || sp->diag_col[sp->diag_ptr[I]] != I)
            return false;
    return true;
}

void check_diagonal_rule(op_mat A) {
    if (A->sparsity->diagonal_elsewhere)
        fail("matrix %s: a format that moves diagonal entries, but sparsity %s reaches a diagonal block through "
             "other indices", A->name.c_str(), A->sparsity->name.c_str());
}

} // namespace

int part_rows(op_mat A) { return A->sparsity->rows->size * A->dim; }

int part_cols(op_mat A, Part part) {
    op_sparsity sp = A->sparsity;
    return (part == Part::diag ? sp->cols->size : (int)sp->garray.size()) * A->dim;
}

int part_blocks(op_mat A, Part part) {
    return part == Part::diag ? A->sparsity->diag_blocks : A->sparsity->offd_blocks;
}

std::size_t position(const Entry& e, int d, int r, int c, bool diagonal) {
    (void)d;
    const int rule = diagonal ? e[mat::aux] >> mat::rule_shift : mat::plain;
    const std::size_t row = (std::size_t)e[mat::offset] + (std::size_t)r * e[mat::row_stride];
    if (rule == mat::diagonal_first)
        return row + (std::size_t)(c == r ? 0 : c + (c < r)) * e[mat::col_stride];
    if (rule == mat::diagonal_separate)
        return r == c ? (std::size_t)e[mat::col_stride] + r : row + (c - (c > r));
    return row + (std::size_t)c * e[mat::col_stride];
}

ScalarCsr csr_layout(op_mat A, Part part, int region, int offset, Diagonal diagonal, int diagonal_offset) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim, nrows = sp->rows->size;
    const Blocks b = blocks(A, part);
    const bool square = sp->rows == sp->cols;
    const Diagonal rule = b.diag && square ? diagonal : Diagonal::none;
    if (rule != Diagonal::none)
        check_diagonal_rule(A);
    if (rule == Diagonal::first && !diagonal_leads(A))
        fail("matrix %s: diagonal-first rows need sparsity %s ordered for them", A->name.c_str(), sp->name.c_str());

    ScalarCsr out;
    out.ptr.assign((std::size_t)nrows * d + 1, 0);
    out.col.reserve((std::size_t)part_blocks(A, part) * d * d);
    for (int I = 0; I < nrows; I++) {
        const int nb = b.ptr[I + 1] - b.ptr[I];
        const bool apart = rule == Diagonal::separate;
        const int length = nb * d - (apart ? 1 : 0);
        for (int r = 0; r < d; r++) {
            out.ptr[(std::size_t)I * d + r + 1] = out.ptr[(std::size_t)I * d + r] + length;
            for (int k = b.ptr[I]; k < b.ptr[I + 1]; k++) {
                const int J = b.col[k];
                if (J == I && rule == Diagonal::first) {
                    out.col.push_back(J * d + r);
                    for (int c = 0; c < d; c++)
                        if (c != r)
                            out.col.push_back(J * d + c);
                } else
                    for (int c = 0; c < d; c++)
                        if (!(J == I && apart && c == r))
                            out.col.push_back(J * d + c);
            }
        }
        // A row's blocks follow one another, d columns each, one fewer after
        // a diagonal block that keeps its diagonal apart.
        long at = (long)offset + out.ptr[(std::size_t)I * d];
        for (int k = b.ptr[I]; k < b.ptr[I + 1]; k++) {
            const int J = b.col[k];
            if (J == I && rule == Diagonal::first)
                A->layout[b.first + k] = entry(at, length, 1, region, mat::diagonal_first);
            else if (J == I && apart)
                A->layout[b.first + k] =
                    entry(at, length, (int)((long)diagonal_offset + (long)I * d), region, mat::diagonal_separate);
            else
                A->layout[b.first + k] = entry(at, length, 1, region);
            at += d - (J == I && apart ? 1 : 0);
        }
    }
    return out;
}

BlockCsr bcsr_layout(op_mat A, Part part, int region, int offset, Major major, bool separate_diagonal,
                     int diagonal_offset) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim, nrows = sp->rows->size, bs = d * d;
    const Blocks b = blocks(A, part);
    const bool apart = separate_diagonal && b.diag && sp->rows == sp->cols;
    const int rs = major == Major::row ? d : 1, cs = major == Major::row ? 1 : d;

    BlockCsr out;
    out.ptr.assign((std::size_t)nrows + 1, 0);
    long next = 0;
    for (int I = 0; I < nrows; I++) {
        for (int k = b.ptr[I]; k < b.ptr[I + 1]; k++) {
            const int J = b.col[k];
            if (apart && J == I) {
                A->layout[b.first + k] = entry((long)diagonal_offset + (long)I * bs, rs, cs, region);
                continue;
            }
            A->layout[b.first + k] = entry((long)offset + next * bs, rs, cs, region);
            out.col.push_back(J);
            next++;
        }
        out.ptr[I + 1] = (int)out.col.size();
    }
    return out;
}

Ell ell_layout(op_mat A, Part part, int region, int offset, int pad) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim, nrows = sp->rows->size, rows = nrows * d;
    const Blocks b = blocks(A, part);

    Ell out;
    for (int I = 0; I < nrows; I++)
        out.width = std::max(out.width, (b.ptr[I + 1] - b.ptr[I]) * d);
    out.col.assign((std::size_t)out.width * rows, pad);
    for (int I = 0; I < nrows; I++)
        for (int k = b.ptr[I]; k < b.ptr[I + 1]; k++) {
            const int kb = k - b.ptr[I], J = b.col[k];
            for (int r = 0; r < d; r++)
                for (int c = 0; c < d; c++)
                    out.col[(std::size_t)(kb * d + c) * rows + I * d + r] = J * d + c;
            A->layout[b.first + k] = entry((long)offset + (long)kb * d * rows + (long)I * d, 1, rows, region);
        }
    return out;
}

Sell sell_layout(op_mat A, Part part, int region, int offset, int slice, int factor, int pad) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim, nrows = sp->rows->size, rows = nrows * d;
    const Blocks b = blocks(A, part);
    if (slice % d != 0)
        fail("matrix %s: SELL-P slices of %d rows split its %d-row blocks", A->name.c_str(), slice, d);

    Sell out;
    out.slice = slice;
    const int slices = (rows + slice - 1) / slice;
    out.lengths.assign(slices, 0);
    out.sets.assign((std::size_t)slices + 1, 0);
    for (int I = 0; I < nrows; I++) {
        int& length = out.lengths[(I * d) / slice];
        length = std::max(length, (b.ptr[I + 1] - b.ptr[I]) * d);
    }
    for (int s = 0; s < slices; s++) {
        out.lengths[s] = (out.lengths[s] + factor - 1) / factor * factor;
        out.sets[s + 1] = out.sets[s] + out.lengths[s];
    }
    out.col.assign((std::size_t)out.sets[slices] * slice, pad);
    for (int I = 0; I < nrows; I++) {
        const int i0 = I * d, s = i0 / slice;
        for (int k = b.ptr[I]; k < b.ptr[I + 1]; k++) {
            const int kb = k - b.ptr[I], J = b.col[k];
            for (int r = 0; r < d; r++)
                for (int c = 0; c < d; c++)
                    out.col[((std::size_t)out.sets[s] + kb * d + c) * slice + (i0 + r) % slice] = J * d + c;
            A->layout[b.first + k] =
                entry((long)offset + ((long)out.sets[s] + (long)kb * d) * slice + i0 % slice, 1, slice, region);
        }
    }
    return out;
}

BlockEll bell_layout(op_mat A, Part part, int region, int offset, Major major, bool interleaved, int pad) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim, nrows = sp->rows->size, bs = d * d;
    const Blocks b = blocks(A, part);

    BlockEll out;
    for (int I = 0; I < nrows; I++)
        out.width = std::max(out.width, b.ptr[I + 1] - b.ptr[I]);
    out.col.assign((std::size_t)out.width * nrows, pad);
    for (int I = 0; I < nrows; I++)
        for (int k = b.ptr[I]; k < b.ptr[I + 1]; k++) {
            const long kb = k - b.ptr[I];
            out.col[(std::size_t)kb * nrows + I] = b.col[k];
            if (interleaved) {
                const int rs = major == Major::row ? d * nrows : nrows, cs = major == Major::row ? nrows : d * nrows;
                A->layout[b.first + k] = entry((long)offset + kb * bs * nrows + I, rs, cs, region);
            } else {
                const int rs = major == Major::row ? d : 1, cs = major == Major::row ? 1 : d;
                A->layout[b.first + k] = entry((long)offset + (kb * nrows + I) * bs, rs, cs, region);
            }
        }
    return out;
}

void csr_located(op_mat A, Part part, int region, const std::vector<int>& ptr, const std::vector<int>& col) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim, nrows = sp->rows->size;
    const Blocks b = blocks(A, part);

    // Where scalar (i, j) is, or -1.
    auto find = [&](int i, int j) -> long {
        auto begin = col.begin() + ptr[i], end = col.begin() + ptr[i + 1];
        auto at = std::find(begin, end, j);
        return at == end ? -1 : (long)(at - col.begin());
    };
    for (int I = 0; I < nrows; I++)
        for (int k = b.ptr[I]; k < b.ptr[I + 1]; k++) {
            const int J = b.col[k];
            const long first = find(I * d, J * d);
            const long rs = d > 1 ? find(I * d + 1, J * d) - first : d;
            const long cs = d > 1 ? find(I * d, J * d + 1) - first : 1;
            if (first < 0)
                fail("matrix %s: the library's storage has no entry (%d, %d)", A->name.c_str(), I * d, J * d);
            for (int r = 0; r < d; r++)
                for (int c = 0; c < d; c++)
                    if (find(I * d + r, J * d + c) != first + r * rs + c * cs)
                        fail("matrix %s: the library keeps block (%d, %d) unevenly", A->name.c_str(), I, J);
            A->layout[b.first + k] = entry(first, (int)rs, (int)cs, region);
        }
}

} // namespace op::solve
