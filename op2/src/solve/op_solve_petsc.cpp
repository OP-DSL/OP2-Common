// The PETSc backend: an AIJ matrix whose parts' values the loops write in
// place on the host, OP2's region being their arrays; on the device a copy
// in COO order handed over at each assembly (MatSetValuesCOO), which "coo"
// asks for on the host too.  "aij:owned" borrows PETSc's own arrays instead
// (AIJCUSPARSE's on the device, experimental).  KSP over vectors wrapping the
// dats; PCSHELL and MATSHELL for the callbacks.

#include <op2/solve/solve.h>

#include <op_lib_c.h>

#include <petscksp.h>
#include <petsc/private/petscimpl.h>

#if OP2_SOLVE_DEVICE
#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#endif

#include <algorithm>
#include <sstream>
#include <vector>

namespace op::solve {

namespace {

#define PETSC_CALL(call) PetscCallAbort(PETSC_COMM_WORLD, call)

class PetscMatrix final : public Matrix {
public:
    PetscMatrix(op_mat A, const std::string& format) : m_A{A} {
        const idx_g_t global = std::max(A->sparsity->global_rows, A->sparsity->global_cols) * A->dim;
        if (global > PETSC_INT_MAX)
            fail("PETSc: matrix %s has %lld global rows or columns, past its %zu-byte PetscInt", A->name.c_str(),
                 (long long)global, sizeof(PetscInt));
        // On the device PETSc's arrays move under it, so it copies by default.
        m_coo = format == "coo" || (on_device && (format.empty() || format == "aij"));
        if (!m_coo && !format.empty() && format != "aij" && format != "aij:owned")
            fail("PETSc: matrix %s: unknown format '%s'", A->name.c_str(), format.c_str());
        if (m_coo)
            coo();
        else if (format == "aij:owned")
            owned();
        else {
            m_views = true;
            views();
        }
    }

    ~PetscMatrix() override { MatDestroy(&m_mat); }

    void assemble() override {
        if (m_coo) {
            PETSC_CALL(MatSetValuesCOO(m_mat, (const PetscScalar *)m_A->regions[0].data, INSERT_VALUES));
            return;
        }
#if OP2_SOLVE_DEVICE
        // The loops wrote PETSc's device arrays: the device copy is the
        // current one, and the matrix has a new state.  A host assembly would
        // let PETSc copy its stale host values over them.
        if (!m_views) {
            for (int p = 0; p < 2; p++) {
                if (m_parts[p] == nullptr)
                    continue;
                PetscScalar *values;
                PETSC_CALL(MatSeqAIJCUSPARSEGetArrayWrite(m_parts[p], &values));
                PETSC_CALL(MatSeqAIJCUSPARSERestoreArrayWrite(m_parts[p], &values));
            }
            PETSC_CALL(PetscObjectStateIncrease((PetscObject)m_mat));
            return;
        }
#endif
        // The arrays are PETSc's own: an assembly tells it they changed.
        PETSC_CALL(MatAssemblyBegin(m_mat, MAT_FINAL_ASSEMBLY));
        PETSC_CALL(MatAssemblyEnd(m_mat, MAT_FINAL_ASSEMBLY));
    }

    // PETSc's own device arrays must not have moved since they were located.
    void zero() override {
#if OP2_SOLVE_DEVICE
        for (int p = 0; p < 2 && !m_views; p++) {
            if (m_parts[p] == nullptr)
                continue;
            PetscScalar *values;
            PETSC_CALL(MatSeqAIJCUSPARSEGetArrayWrite(m_parts[p], &values));
            PETSC_CALL(MatSeqAIJCUSPARSERestoreArrayWrite(m_parts[p], &values));
            if ((char *)values != m_A->regions[p].data)
                fail("PETSc: matrix %s: its device values moved", m_A->name.c_str());
        }
#endif
    }

    Mat mat() const { return m_mat; }

private:
    // The diagonal and off-diagonal parts' arrays over one region of values,
    // which the loops write in place.
    void views() {
        static_assert(sizeof(PetscInt) == sizeof(int), "PETSc's indices must be int");
        op_sparsity sp = m_A->sparsity;
        const PetscInt d = m_A->dim, m = sp->rows->size * d;
        const std::size_t nd = (std::size_t)part_blocks(m_A, Part::diag) * d * d;
        const std::size_t no = (std::size_t)part_blocks(m_A, Part::offd) * d * d;
        PetscScalar *data = (PetscScalar *)lend(m_A, 0, (nd + no) * sizeof(PetscScalar));
        m_diag = csr_layout(m_A, Part::diag, 0, 0);
        m_offd = csr_layout(m_A, Part::offd, 0, (int)nd);

        PetscInt *garray;
        PETSC_CALL(PetscMalloc1(sp->garray.size() * d + 1, &garray));
        for (std::size_t k = 0; k < sp->garray.size(); k++)
            for (PetscInt c = 0; c < d; c++)
                garray[k * d + c] = (PetscInt)(sp->garray[k] * d + c);

        Mat diag, offd;
        PETSC_CALL(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, sp->cols->size * d, m_diag.ptr.data(),
                                             m_diag.col.data(), data, &diag));
        PETSC_CALL(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, (PetscInt)sp->garray.size() * d,
                                             m_offd.ptr.data(), m_offd.col.data(), data + nd, &offd));
        PETSC_CALL(MatCreateMPIAIJWithSeqAIJ(comm(), sp->global_rows * d, sp->global_cols * d, diag, offd, garray,
                                             &m_mat));
        PETSC_CALL(MatSetBlockSize(m_mat, d));
    }

    // PETSc's own parts on the pattern (AIJCUSPARSE on the device), which the
    // loops write in place: their values borrowed as regions 0 and 1, each
    // block found from PETSc's own row pointers and columns.
    void owned() {
        op_sparsity sp = m_A->sparsity;
        const PetscInt d = m_A->dim, nrows = sp->rows->size, m = nrows * d;
        std::vector<PetscInt> ptr(m + 1, 0), col;
        std::vector<PetscInt> row;
        for (int I = 0; I < nrows; I++)
            for (PetscInt r = 0; r < d; r++) {
                row.clear();
                for (int k = sp->diag_ptr[I]; k < sp->diag_ptr[I + 1]; k++)
                    for (PetscInt c = 0; c < d; c++)
                        row.push_back((PetscInt)((sp->col_start + sp->diag_col[k]) * d + c));
                for (int k = sp->offd_ptr[I]; k < sp->offd_ptr[I + 1]; k++)
                    for (PetscInt c = 0; c < d; c++)
                        row.push_back((PetscInt)(sp->garray[sp->offd_col[k]] * d + c));
                std::sort(row.begin(), row.end());
                col.insert(col.end(), row.begin(), row.end());
                ptr[I * d + r + 1] = (PetscInt)col.size();
            }
        PETSC_CALL(MatCreate(comm(), &m_mat));
        PETSC_CALL(MatSetSizes(m_mat, m, sp->cols->size * d, sp->global_rows * d, sp->global_cols * d));
        PETSC_CALL(MatSetType(m_mat, on_device ? MATMPIAIJCUSPARSE : MATMPIAIJ));
        PETSC_CALL(MatSetBlockSize(m_mat, d));
        PETSC_CALL(MatMPIAIJSetPreallocationCSR(m_mat, ptr.data(), col.data(), nullptr));

        const PetscInt *colmap;
        PETSC_CALL(MatMPIAIJGetSeqAIJ(m_mat, &m_parts[0], &m_parts[1], &colmap));
        for (int p = 0; p < 2; p++) {
            PetscInt rows, cols;
            PETSC_CALL(MatGetSize(m_parts[p], &rows, &cols));
            std::vector<int> hptr(rows + 1), hcol;
            PetscScalar *values = nullptr;
#if OP2_SOLVE_DEVICE
            const PetscInt *dptr, *dcol;
            PETSC_CALL(MatSeqAIJCUSPARSEGetIJ(m_parts[p], PETSC_FALSE, &dptr, &dcol));
            GPU_CHECK(gpuMemcpy(hptr.data(), dptr, sizeof(int) * (rows + 1), gpuMemcpyDeviceToHost));
            hcol.resize(hptr[rows]);
            GPU_CHECK(gpuMemcpy(hcol.data(), dcol, sizeof(int) * hcol.size(), gpuMemcpyDeviceToHost));
            PETSC_CALL(MatSeqAIJCUSPARSERestoreIJ(m_parts[p], PETSC_FALSE, &dptr, &dcol));
            PETSC_CALL(MatSeqAIJCUSPARSEGetArrayWrite(m_parts[p], &values));
            PETSC_CALL(MatSeqAIJCUSPARSERestoreArrayWrite(m_parts[p], &values));
#else
            const PetscInt *hi, *hj;
            PetscMemType type;
            PETSC_CALL(MatSeqAIJGetCSRAndMemType(m_parts[p], &hi, &hj, &values, &type));
            std::copy(hi, hi + rows + 1, hptr.begin());
            hcol.assign(hj, hj + hptr[rows]);
#endif
            if (hcol.empty()) {
                m_parts[p] = nullptr;
                continue;
            }
            // The off-diagonal part's columns into PETSc's colmap, as ours
            // into the sparsity's garray.
            if (p == 1)
                for (int& c : hcol) {
                    const idx_g_t global = colmap[c];
                    auto at = std::lower_bound(sp->garray.begin(), sp->garray.end(), global / d);
                    c = (int)(at - sp->garray.begin()) * d + (int)(global % d);
                }
            csr_located(m_A, p == 0 ? Part::diag : Part::offd, p, hptr, hcol);
            borrow(m_A, p, values, hcol.size() * sizeof(PetscScalar));
        }
    }

    // A region of values in the order of PETSc's COO entries: the diagonal
    // part's blocks row by row, then the off-diagonal part's, each row major.
    void coo() {
        op_sparsity sp = m_A->sparsity;
        const PetscInt d = m_A->dim;
        const std::size_t entries = (std::size_t)(sp->diag_blocks + sp->offd_blocks) * d * d;
        lend(m_A, 0, entries * sizeof(PetscScalar));
        bcsr_layout(m_A, Part::diag, 0, 0, Major::row);
        bcsr_layout(m_A, Part::offd, 0, (int)((std::size_t)sp->diag_blocks * d * d), Major::row);

        PETSC_CALL(MatCreate(comm(), &m_mat));
        PETSC_CALL(MatSetSizes(m_mat, sp->rows->size * d, sp->cols->size * d, sp->global_rows * d,
                               sp->global_cols * d));
        PETSC_CALL(MatSetType(m_mat, on_device ? MATAIJCUSPARSE : MATAIJ));
        PETSC_CALL(MatSetBlockSize(m_mat, d));

        std::vector<PetscInt> rows, cols;
        rows.reserve(entries);
        cols.reserve(entries);
        auto block = [&](idx_g_t I, idx_g_t J) {
            for (PetscInt r = 0; r < d; r++)
                for (PetscInt c = 0; c < d; c++) {
                    rows.push_back((PetscInt)(I * d + r));
                    cols.push_back((PetscInt)(J * d + c));
                }
        };
        const int nrows = sp->rows->size;
        for (int I = 0; I < nrows; I++)
            for (int k = sp->diag_ptr[I]; k < sp->diag_ptr[I + 1]; k++)
                block(sp->row_start + I, sp->col_start + sp->diag_col[k]);
        for (int I = 0; I < nrows; I++)
            for (int k = sp->offd_ptr[I]; k < sp->offd_ptr[I + 1]; k++)
                block(sp->row_start + I, sp->garray[sp->offd_col[k]]);
        PETSC_CALL(MatSetPreallocationCOO(m_mat, (PetscCount)entries, rows.data(), cols.data()));
    }

    op_mat m_A;
    Mat m_mat = nullptr;
    bool m_coo = false, m_views = false;
    Mat m_parts[2] = {nullptr, nullptr};
    ScalarCsr m_diag, m_offd;
};

Mat mat_of(op_mat A) { return static_cast<PetscMatrix *>(A->lib.get())->mat(); }

// A dat's owned values as a library vector, and back.
void place(Vec v, op_dat dat) {
    if (on_device)
        PETSC_CALL(VecCUDAPlaceArray(v, (const PetscScalar *)memory(dat)));
    else
        PETSC_CALL(VecPlaceArray(v, (const PetscScalar *)memory(dat)));
}

void reset(Vec v) {
    if (on_device)
        PETSC_CALL(VecCUDAResetArray(v));
    else
        PETSC_CALL(VecResetArray(v));
}

// The library's vectors over the device or host arrays a callback runs on.
PetscErrorCode through(op_solver s, op_apply_fn apply, void *ctx, Vec in, Vec out) {
    const PetscScalar *r;
    PetscScalar *z;
    PetscFunctionBeginUser;
    if (on_device) {
        PetscCall(VecCUDAGetArrayRead(in, &r));
        PetscCall(VecCUDAGetArrayWrite(out, &z));
    } else {
        PetscCall(VecGetArrayRead(in, &r));
        PetscCall(VecGetArrayWrite(out, &z));
    }
    apply_through(s, apply, ctx, r, z);
    if (on_device) {
        PetscCall(VecCUDARestoreArrayRead(in, &r));
        PetscCall(VecCUDARestoreArrayWrite(out, &z));
    } else {
        PetscCall(VecRestoreArrayRead(in, &r));
        PetscCall(VecRestoreArrayWrite(out, &z));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode pc_apply(PC pc, Vec r, Vec z) {
    op_solver s;
    PetscFunctionBeginUser;
    PetscCall(PCShellGetContext(pc, &s));
    PetscCall(through(s, s->pc, s->pc_ctx, r, z));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode op_mult(Mat A, Vec x, Vec y) {
    op_solver s;
    PetscFunctionBeginUser;
    PetscCall(MatShellGetContext(A, &s));
    PetscCall(through(s, s->op, s->op_ctx, x, y));
    PetscFunctionReturn(PETSC_SUCCESS);
}

KSPType ksp_type(const std::string& method) {
    if (method == "cg")
        return KSPCG;
    if (method == "gmres")
        return KSPGMRES;
    if (method == "fgmres")
        return KSPFGMRES;
    if (method == "bicgstab")
        return KSPBCGS;
    fail("PETSc: unknown method '%s'", method.c_str());
}

class PetscSolver final : public Solver {
public:
    explicit PetscSolver(op_solver s) : m_s{s} {
        const Options& o = s->options;
        op_sparsity sp = s->A->sparsity;
        const PetscInt d = s->A->dim, n = sp->rows->size * d, N = sp->global_rows * d;
        PETSC_CALL(KSPCreate(comm(), &m_ksp));

        Mat A = mat_of(s->A), P = s->P != nullptr ? mat_of(s->P) : A;
        if (s->op != nullptr) {
            PETSC_CALL(MatCreateShell(comm(), n, n, N, N, s, &m_shell));
            PETSC_CALL(MatShellSetOperation(m_shell, MATOP_MULT, (void (*)(void))op_mult));
            if (on_device)
                PETSC_CALL(MatShellSetVecType(m_shell, VECCUDA));
            A = m_shell;
        }
        PETSC_CALL(KSPSetOperators(m_ksp, A, P));
        PETSC_CALL(KSPSetType(m_ksp, ksp_type(o.method)));
        PETSC_CALL(KSPSetTolerances(m_ksp, o.rtol, o.atol, PETSC_DETERMINE, o.maxit));
        if (o.method == "gmres" || o.method == "fgmres")
            PETSC_CALL(KSPGMRESSetRestart(m_ksp, o.restart));
        PETSC_CALL(KSPSetInitialGuessNonzero(m_ksp, PETSC_TRUE));
        // Converge on the true residual, ||b - A x|| / ||b||, as every backend does.
        if (o.method == "cg")
            PETSC_CALL(KSPSetNormType(m_ksp, KSP_NORM_UNPRECONDITIONED));
        else
            PETSC_CALL(KSPSetPCSide(m_ksp, PC_RIGHT));

        PC pc;
        PETSC_CALL(KSPGetPC(m_ksp, &pc));
        int ranks;
        MPI_Comm_size(comm(), &ranks);
        if (o.pc == "none")
            PETSC_CALL(PCSetType(pc, PCNONE));
        else if (o.pc == "jacobi")
            PETSC_CALL(PCSetType(pc, d > 1 ? PCPBJACOBI : PCJACOBI));
        else if (o.pc == "bjacobi" || (o.pc == "ilu" && ranks > 1))
            PETSC_CALL(PCSetType(pc, PCBJACOBI));
        else if (o.pc == "ilu")
            PETSC_CALL(PCSetType(pc, PCILU));
        else if (o.pc == "amg")
            PETSC_CALL(PCSetType(pc, PCGAMG));
        else if (o.pc == "custom") {
            PETSC_CALL(PCSetType(pc, PCSHELL));
            PETSC_CALL(PCShellSetApply(pc, pc_apply));
            PETSC_CALL(PCShellSetContext(pc, s));
        } else
            fail("PETSc: unknown pc '%s'", o.pc.c_str());

        // The solver's own library options under its prefix, the rest as given.
        std::string prefix = s->name + "_";
        PETSC_CALL(KSPSetOptionsPrefix(m_ksp, prefix.c_str()));
        std::istringstream words(o.library);
        std::string word, options;
        while (words >> word) {
            bool own = false;
            for (const char *kind : {"-ksp_", "-pc_", "-mg_", "-sub_"})
                own |= word.rfind(kind, 0) == 0;
            options += (own ? "-" + prefix + word.substr(1) : word) + " ";
        }
        if (o.monitor)
            options += "-" + prefix + "ksp_monitor ";
        PETSC_CALL(PetscOptionsInsertString(nullptr, options.c_str()));
        PETSC_CALL(KSPSetFromOptions(m_ksp));

        if (on_device) {
            PETSC_CALL(VecCreateMPICUDAWithArray(comm(), d, n, N, nullptr, &m_b));
            PETSC_CALL(VecCreateMPICUDAWithArray(comm(), d, n, N, nullptr, &m_x));
        } else {
            PETSC_CALL(VecCreateMPIWithArray(comm(), d, n, N, nullptr, &m_b));
            PETSC_CALL(VecCreateMPIWithArray(comm(), d, n, N, nullptr, &m_x));
        }
    }

    ~PetscSolver() override {
        VecDestroy(&m_b);
        VecDestroy(&m_x);
        MatDestroy(&m_shell);
        KSPDestroy(&m_ksp);
    }

    bool solve(op_dat b, op_dat x) override {
        place(m_b, b);
        place(m_x, x);
        PETSC_CALL(VecNorm(m_b, NORM_2, &m_bnorm));
        PETSC_CALL(KSPSolve(m_ksp, m_b, m_x));
        reset(m_b);
        reset(m_x);

        PetscInt its;
        PetscReal norm;
        KSPConvergedReason reason;
        PETSC_CALL(KSPGetIterationNumber(m_ksp, &its));
        PETSC_CALL(KSPGetResidualNorm(m_ksp, &norm));
        PETSC_CALL(KSPGetConvergedReason(m_ksp, &reason));
        iterations = (int)its;
        residual = m_bnorm > 0 ? (double)norm / m_bnorm : (double)norm;
        return reason > 0;
    }

private:
    op_solver m_s;
    KSP m_ksp = nullptr;
    Mat m_shell = nullptr;
    Vec m_b = nullptr, m_x = nullptr;
    PetscReal m_bnorm = 0;
};

class PetscLibrary final : public Library {
public:
    PetscLibrary(int argc, char **argv) {
        PETSC_COMM_WORLD = comm();
        if (on_device && !OP_gpu_direct)
            PETSC_CALL(PetscOptionsSetValue(nullptr, "-use_gpu_aware_mpi", "0"));
        PETSC_CALL(PetscInitialize(&argc, &argv, nullptr, nullptr));
        // The command line is the application's too.
        PETSC_CALL(PetscOptionsSetValue(nullptr, "-options_left", "0"));
    }

    ~PetscLibrary() override { PetscFinalize(); }

    Order order() const override { return Order::sorted; }

    std::unique_ptr<Matrix> matrix(op_mat A, const std::string& format) override {
        return std::make_unique<PetscMatrix>(A, format);
    }

    std::unique_ptr<Solver> solver(op_solver s) override { return std::make_unique<PetscSolver>(s); }
};

} // namespace

std::unique_ptr<Library> make_petsc(int argc, char **argv) {
    return std::make_unique<PetscLibrary>(argc, argv);
}

} // namespace op::solve
