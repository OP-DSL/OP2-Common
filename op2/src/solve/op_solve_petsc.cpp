// The PETSc backend: on the host an AIJ matrix whose parts' arrays are the
// op_mat's values, written in place by the loops; on the device an
// AIJCUSPARSE matrix assembled through PETSc's COO interface from the values
// in place there; KSP over vectors wrapping the dats.

#include <op2/solve/solve.h>

#include <op_lib_c.h>

#include <petscksp.h>

#include <sstream>
#include <vector>

namespace op::solve {

namespace {

#define PETSC_CALL(call) PetscCallAbort(PETSC_COMM_WORLD, call)

class PetscMatrix final : public Matrix {
public:
    explicit PetscMatrix(op_mat A) : m_A{A} {
        if (on_device)
            coo();
        else
            views();
    }

    ~PetscMatrix() override {
        MatDestroy(&m_mat);
        if (m_to != nullptr) {
            release(m_to);
            PetscFree(m_staged);
        }
    }

    void assemble() override {
        if (on_device) {
            PETSC_CALL(MatSetValuesCOO(m_mat, (const PetscScalar *)memory(m_A->values), INSERT_VALUES));
            return;
        }
        // The arrays are PETSc's own: an assembly tells it they changed.
        if (m_to != nullptr)
            scatter(m_staged, (const double *)memory(m_A->values), m_to, m_entries);
        PETSC_CALL(MatAssemblyBegin(m_mat, MAT_FINAL_ASSEMBLY));
        PETSC_CALL(MatAssemblyEnd(m_mat, MAT_FINAL_ASSEMBLY));
    }

    Mat mat() const { return m_mat; }

private:
    // The diagonal and off-diagonal parts over the values (or, blocked, a copy
    // permuted from them), their columns as the layout's scalar CSR has them.
    void views() {
        static_assert(sizeof(PetscInt) == sizeof(int), "PETSc's indices must be int");
        op_sparsity sp = m_A->sparsity;
        const Scalar& sc = m_A->scalar;
        const PetscInt d = m_A->dim, m = sp->rows->size * d;
        const std::size_t nd = sc.diag.col.size(), no = sc.offd.col.size();

        PetscScalar *data = (PetscScalar *)memory(m_A->values);
        if (m_A->mode != Mode::direct) {
            PETSC_CALL(PetscMalloc1(nd + no + 1, &m_staged));
            data = m_staged;
            std::vector<int> to((std::size_t)(sp->diag_blocks + sp->offd_blocks) * d * d);
            for (std::size_t k = 0; k < to.size(); k++)
                to[k] = position(sc, d, (int)(k / (d * d)), (int)(k % (d * d)));
            m_to = upload(to);
            m_entries = to.size();
        }

        PetscInt *garray;
        PETSC_CALL(PetscMalloc1(sp->garray.size() * d + 1, &garray));
        for (std::size_t k = 0; k < sp->garray.size(); k++)
            for (PetscInt c = 0; c < d; c++)
                garray[k * d + c] = (PetscInt)(sp->garray[k] * d + c);

        Mat diag, offd;
        PETSC_CALL(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, sp->cols->size * d, (PetscInt *)sc.diag.ptr.data(),
                                             (PetscInt *)sc.diag.col.data(), data, &diag));
        PETSC_CALL(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, (PetscInt)sp->garray.size() * d,
                                             (PetscInt *)sc.offd.ptr.data(), (PetscInt *)sc.offd.col.data(),
                                             data + nd, &offd));
        PETSC_CALL(MatCreateMPIAIJWithSeqAIJ(comm(), sp->global_rows * d, sp->global_cols * d, diag, offd, garray,
                                             &m_mat));
        PETSC_CALL(MatSetBlockSize(m_mat, d));
    }

    void coo() {
        op_sparsity sp = m_A->sparsity;
        const PetscInt d = m_A->dim;
        PETSC_CALL(MatCreate(comm(), &m_mat));
        PETSC_CALL(MatSetSizes(m_mat, sp->rows->size * d, sp->cols->size * d, sp->global_rows * d,
                               sp->global_cols * d));
        PETSC_CALL(MatSetType(m_mat, on_device ? MATAIJCUSPARSE : MATAIJ));
        PETSC_CALL(MatSetBlockSize(m_mat, d));

        // One COO entry per value, in the values' order: the diagonal part's
        // blocks row by row, then the off-diagonal part's, each row major.
        std::vector<PetscInt> rows, cols;
        const std::size_t entries = (std::size_t)(sp->diag_blocks + sp->offd_blocks) * d * d;
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
    PetscScalar *m_staged = nullptr;
    int *m_to = nullptr;
    std::size_t m_entries = 0;
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

    // On the host the values are PETSc's arrays; on the device it copies them.
    bool direct() const override { return !on_device; }

    std::unique_ptr<Matrix> matrix(op_mat A) override { return std::make_unique<PetscMatrix>(A); }

    std::unique_ptr<Solver> solver(op_solver s) override { return std::make_unique<PetscSolver>(s); }
};

} // namespace

std::unique_ptr<Library> make_petsc(int argc, char **argv) {
    return std::make_unique<PetscLibrary>(argc, argv);
}

} // namespace op::solve
