// The hypre backend: a ParCSR matrix whose diagonal and off-diagonal parts'
// data are the op_mat's values, written in place by the loops (or, for blocks
// written by untranslated loops, a copy permuted from them), Krylov solvers
// over ParVectors wrapping the dats.

#include <op2/solve/solve.h>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <_hypre_parcsr_mv.h>

#include <cstdlib>
#include <vector>

namespace op::solve {

namespace {

#define HYPRE_CALL(call)                                                                              \
    do {                                                                                              \
        HYPRE_Int error_ = (call);                                                                    \
        if (error_ != 0 && !HYPRE_CheckError(error_, HYPRE_ERROR_CONV))                              \
            fail("hypre error %d at %s:%d", (int)error_, __FILE__, __LINE__);                       \
    } while (0)

constexpr HYPRE_MemoryLocation location = on_device ? HYPRE_MEMORY_DEVICE : HYPRE_MEMORY_HOST;

template<typename T>
T *to_location(const std::vector<T>& host) {
    T *dst = hypre_TAlloc(T, host.size() + 1, location);
    hypre_TMemcpy(dst, host.data(), T, host.size(), location, HYPRE_MEMORY_HOST);
    return dst;
}

class HypreMatrix final : public Matrix {
public:
    explicit HypreMatrix(op_mat A) : m_A{A} {
        op_sparsity sp = A->sparsity;
        const int d = A->dim, nrows = sp->rows->size;
        HYPRE_BigInt rows[2] = {(HYPRE_BigInt)(sp->row_start * d), (HYPRE_BigInt)((sp->row_start + nrows) * d)};
        HYPRE_BigInt cols[2] = {(HYPRE_BigInt)(sp->col_start * d),
                                (HYPRE_BigInt)((sp->col_start + sp->cols->size) * d)};

        const Scalar& sc = A->scalar;
        std::vector<HYPRE_Int> diag_i(sc.diag.ptr.begin(), sc.diag.ptr.end()), diag_j(sc.diag.col.begin(), sc.diag.col.end());
        std::vector<HYPRE_Int> offd_i(sc.offd.ptr.begin(), sc.offd.ptr.end()), offd_j(sc.offd.col.begin(), sc.offd.col.end());
        std::vector<HYPRE_BigInt> col_map(sp->garray.size() * d);
        for (std::size_t k = 0; k < sp->garray.size(); k++)
            for (int c = 0; c < d; c++)
                col_map[k * d + c] = (HYPRE_BigInt)(sp->garray[k] * d + c);

        m_par = hypre_ParCSRMatrixCreate(comm(), (HYPRE_BigInt)(sp->global_rows * d),
                                         (HYPRE_BigInt)(sp->global_cols * d), rows, cols, (HYPRE_Int)col_map.size(),
                                         (HYPRE_Int)diag_j.size(), (HYPRE_Int)offd_j.size());
        hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(m_par), *offd = hypre_ParCSRMatrixOffd(m_par);
        hypre_CSRMatrixMemoryLocation(diag) = location;
        hypre_CSRMatrixMemoryLocation(offd) = location;
        hypre_CSRMatrixI(diag) = to_location(diag_i);
        hypre_CSRMatrixI(offd) = to_location(offd_i);
        m_diag_j = to_location(diag_j);
        m_offd_j = to_location(offd_j);
        hypre_CSRMatrixJ(diag) = m_diag_j;
        hypre_CSRMatrixJ(offd) = m_offd_j;

        // In place, the values are hypre's data; blocked, a copy is.
        double *data = (double *)memory(A->values);
        if (A->mode != Mode::direct) {
            m_staged = hypre_CTAlloc(HYPRE_Complex, diag_j.size() + offd_j.size() + 1, location);
            data = m_staged;
            std::vector<int> to((std::size_t)(sp->diag_blocks + sp->offd_blocks) * d * d);
            for (std::size_t k = 0; k < to.size(); k++)
                to[k] = position(sc, d, (int)(k / (d * d)), (int)(k % (d * d)));
            m_to = upload(to);
            m_entries = to.size();
        }
        hypre_CSRMatrixData(diag) = data;
        hypre_CSRMatrixData(offd) = data + diag_j.size();
        hypre_CSRMatrixOwnsData(diag) = 0;
        hypre_CSRMatrixOwnsData(offd) = 0;

        hypre_ParCSRMatrixColMapOffd(m_par) = hypre_TAlloc(HYPRE_BigInt, col_map.size() + 1, HYPRE_MEMORY_HOST);
        hypre_TMemcpy(hypre_ParCSRMatrixColMapOffd(m_par), col_map.data(), HYPRE_BigInt, col_map.size(),
                      HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
        if (on_device)
            hypre_ParCSRMatrixCopyColMapOffdToDevice(m_par);
        hypre_CSRMatrixSetRownnz(diag);
        hypre_CSRMatrixSetRownnz(offd);
        hypre_ParCSRMatrixSetNumNonzeros(m_par);
        hypre_ParCSRMatrixSetDNumNonzeros(m_par);
        hypre_MatvecCommPkgCreate(m_par);
    }

    ~HypreMatrix() override {
        hypre_ParCSRMatrixDestroy(m_par);
        hypre_TFree(m_diag_j, location);
        hypre_TFree(m_offd_j, location);
        hypre_TFree(m_staged, location);
        if (m_to != nullptr)
            release(m_to);
    }

    void assemble() override {
        if (m_to != nullptr)
            scatter(m_staged, (const double *)memory(m_A->values), m_to, m_entries);
    }

    hypre_ParCSRMatrix *par() const { return m_par; }

private:
    op_mat m_A;
    hypre_ParCSRMatrix *m_par = nullptr;
    HYPRE_Int *m_diag_j = nullptr, *m_offd_j = nullptr;
    HYPRE_Complex *m_staged = nullptr;
    int *m_to = nullptr;
    std::size_t m_entries = 0;
};

hypre_ParCSRMatrix *par_of(op_mat A) { return static_cast<HypreMatrix *>(A->lib.get())->par(); }

// A ParVector over n values the caller owns, pointed at a dat per solve.
hypre_ParVector *wrap(op_mat A) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim;
    HYPRE_BigInt part[2] = {(HYPRE_BigInt)(sp->row_start * d), (HYPRE_BigInt)((sp->row_start + sp->rows->size) * d)};
    hypre_ParVector *v = hypre_ParVectorCreate(comm(), (HYPRE_BigInt)(sp->global_rows * d), part);
    hypre_Vector *local = hypre_ParVectorLocalVector(v);
    static double placeholder;
    hypre_VectorData(local) = &placeholder;
    hypre_ParVectorInitialize_v2(v, location);
    hypre_VectorOwnsData(local) = 0;
    return v;
}

void point(hypre_ParVector *v, op_dat dat) { hypre_VectorData(hypre_ParVectorLocalVector(v)) = (double *)memory(dat); }

double *data_of(HYPRE_ParVector v) { return hypre_VectorData(hypre_ParVectorLocalVector((hypre_ParVector *)v)); }

// The callbacks hypre's Krylov solvers take for a preconditioner, given the
// solver as their data.
HYPRE_Int custom_apply(HYPRE_Solver data, HYPRE_ParCSRMatrix, HYPRE_ParVector r, HYPRE_ParVector z) {
    op_solver s = (op_solver)data;
    apply_through(s, s->pc, s->pc_ctx, data_of(r), data_of(z));
    return 0;
}

HYPRE_Int no_setup(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector) { return 0; }

// One Krylov method's entry points.
struct Krylov {
    HYPRE_Int (*create)(MPI_Comm, HYPRE_Solver *);
    HYPRE_Int (*destroy)(HYPRE_Solver);
    HYPRE_Int (*setup)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
    HYPRE_Int (*solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
    HYPRE_Int (*tol)(HYPRE_Solver, HYPRE_Real);
    HYPRE_Int (*atol)(HYPRE_Solver, HYPRE_Real);
    HYPRE_Int (*maxit)(HYPRE_Solver, HYPRE_Int);
    HYPRE_Int (*precond)(HYPRE_Solver, HYPRE_PtrToParSolverFcn, HYPRE_PtrToParSolverFcn, HYPRE_Solver);
    HYPRE_Int (*iterations)(HYPRE_Solver, HYPRE_Int *);
    HYPRE_Int (*residual)(HYPRE_Solver, HYPRE_Real *);
    HYPRE_Int (*restart)(HYPRE_Solver, HYPRE_Int);
};

Krylov krylov(const std::string& method) {
    if (method == "cg")
        return {HYPRE_ParCSRPCGCreate, HYPRE_ParCSRPCGDestroy, HYPRE_ParCSRPCGSetup, HYPRE_ParCSRPCGSolve,
                HYPRE_ParCSRPCGSetTol, HYPRE_ParCSRPCGSetAbsoluteTol, HYPRE_ParCSRPCGSetMaxIter,
                HYPRE_ParCSRPCGSetPrecond, HYPRE_ParCSRPCGGetNumIterations,
                HYPRE_ParCSRPCGGetFinalRelativeResidualNorm, nullptr};
    if (method == "gmres")
        return {HYPRE_ParCSRGMRESCreate, HYPRE_ParCSRGMRESDestroy, HYPRE_ParCSRGMRESSetup, HYPRE_ParCSRGMRESSolve,
                HYPRE_ParCSRGMRESSetTol, HYPRE_ParCSRGMRESSetAbsoluteTol, HYPRE_ParCSRGMRESSetMaxIter,
                HYPRE_ParCSRGMRESSetPrecond, HYPRE_ParCSRGMRESGetNumIterations,
                HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm, HYPRE_ParCSRGMRESSetKDim};
    if (method == "fgmres")
        return {HYPRE_ParCSRFlexGMRESCreate, HYPRE_ParCSRFlexGMRESDestroy, HYPRE_ParCSRFlexGMRESSetup,
                HYPRE_ParCSRFlexGMRESSolve, HYPRE_ParCSRFlexGMRESSetTol, HYPRE_ParCSRFlexGMRESSetAbsoluteTol,
                HYPRE_ParCSRFlexGMRESSetMaxIter, HYPRE_ParCSRFlexGMRESSetPrecond,
                HYPRE_ParCSRFlexGMRESGetNumIterations, HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm,
                HYPRE_ParCSRFlexGMRESSetKDim};
    if (method == "bicgstab")
        return {HYPRE_ParCSRBiCGSTABCreate, HYPRE_ParCSRBiCGSTABDestroy, HYPRE_ParCSRBiCGSTABSetup,
                HYPRE_ParCSRBiCGSTABSolve, HYPRE_ParCSRBiCGSTABSetTol, HYPRE_ParCSRBiCGSTABSetAbsoluteTol,
                HYPRE_ParCSRBiCGSTABSetMaxIter, HYPRE_ParCSRBiCGSTABSetPrecond,
                HYPRE_ParCSRBiCGSTABGetNumIterations, HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm,
                nullptr};
    fail("hypre: unknown method '%s'", method.c_str());
}

class HypreSolver final : public Solver {
public:
    explicit HypreSolver(op_solver s) : m_s{s}, m_krylov{krylov(s->options.method)} {
        const Options& o = s->options;
        if (s->op != nullptr)
            fail("hypre: solver %s: a par-loop operator is not supported", s->name.c_str());

        HYPRE_CALL(m_krylov.create(comm(), &m_solver));
        HYPRE_CALL(m_krylov.tol(m_solver, o.rtol));
        HYPRE_CALL(m_krylov.atol(m_solver, o.atol));
        HYPRE_CALL(m_krylov.maxit(m_solver, o.maxit));
        if (m_krylov.restart != nullptr)
            HYPRE_CALL(m_krylov.restart(m_solver, o.restart));
        if (o.method == "cg")
            HYPRE_CALL(HYPRE_ParCSRPCGSetTwoNorm(m_solver, 1));

        if (o.pc == "amg") {
            HYPRE_CALL(HYPRE_BoomerAMGCreate(&m_pc));
            HYPRE_CALL(HYPRE_BoomerAMGSetMaxIter(m_pc, 1));
            HYPRE_CALL(HYPRE_BoomerAMGSetTol(m_pc, 0.0));
            HYPRE_CALL(HYPRE_BoomerAMGSetCoarsenType(m_pc, 8));
            HYPRE_CALL(HYPRE_BoomerAMGSetInterpType(m_pc, 6));
            HYPRE_CALL(HYPRE_BoomerAMGSetRelaxType(m_pc, on_device ? 18 : 8));
            HYPRE_CALL(HYPRE_BoomerAMGSetNumFunctions(m_pc, s->A->dim));
            m_apply = HYPRE_BoomerAMGSolve;
            m_setup = HYPRE_BoomerAMGSetup;
            m_pc_destroy = HYPRE_BoomerAMGDestroy;
        } else if (o.pc == "ilu") {
            HYPRE_CALL(HYPRE_ILUCreate(&m_pc));
            HYPRE_CALL(HYPRE_ILUSetMaxIter(m_pc, 1));
            HYPRE_CALL(HYPRE_ILUSetTol(m_pc, 0.0));
            m_apply = HYPRE_ILUSolve;
            m_setup = HYPRE_ILUSetup;
            m_pc_destroy = HYPRE_ILUDestroy;
        } else if (o.pc == "jacobi") {
            m_apply = HYPRE_ParCSRDiagScale;
            m_setup = HYPRE_ParCSRDiagScaleSetup;
        } else if (o.pc == "custom") {
            m_apply = custom_apply;
            m_pc = (HYPRE_Solver)s;
        } else if (o.pc != "none")
            fail("hypre: unknown pc '%s'", o.pc.c_str());
        if (m_apply != nullptr)
            HYPRE_CALL(m_krylov.precond(m_solver, m_apply, no_setup, m_pc));

        m_b = wrap(s->A);
        m_x = wrap(s->A);
    }

    ~HypreSolver() override {
        m_krylov.destroy(m_solver);
        if (m_pc_destroy != nullptr)
            m_pc_destroy(m_pc);
        hypre_ParVectorDestroy(m_b);
        hypre_ParVectorDestroy(m_x);
    }

    bool solve(op_dat b, op_dat x) override {
        point(m_b, b);
        point(m_x, x);
        auto A = (HYPRE_ParCSRMatrix)par_of(m_s->A);
        auto bv = (HYPRE_ParVector)m_b, xv = (HYPRE_ParVector)m_x;

        // The preconditioner is built from P when there is one, and again
        // whenever its matrix has been assembled since.
        op_mat P = m_s->P != nullptr ? m_s->P : m_s->A;
        if (!m_ready) {
            HYPRE_CALL(m_krylov.setup(m_solver, A, bv, xv));
            m_ready = true;
        }
        if (m_setup != nullptr && m_pc_state != P->state) {
            HYPRE_CALL(m_setup(m_pc, (HYPRE_ParCSRMatrix)par_of(P), bv, xv));
            m_pc_state = P->state;
        }
        HYPRE_CALL(m_krylov.solve(m_solver, A, bv, xv));
        HYPRE_ClearAllErrors();

        HYPRE_Int its;
        HYPRE_Real norm;
        HYPRE_CALL(m_krylov.iterations(m_solver, &its));
        HYPRE_CALL(m_krylov.residual(m_solver, &norm));
        iterations = (int)its;
        residual = (double)norm;
        return norm <= m_s->options.rtol || its < m_s->options.maxit;
    }

private:
    op_solver m_s;
    Krylov m_krylov;
    HYPRE_Solver m_solver = nullptr, m_pc = nullptr;
    HYPRE_PtrToParSolverFcn m_apply = nullptr, m_setup = nullptr;
    HYPRE_Int (*m_pc_destroy)(HYPRE_Solver) = nullptr;
    hypre_ParVector *m_b = nullptr, *m_x = nullptr;
    bool m_ready = false;
    int m_pc_state = -1;
};

class HypreLibrary final : public Library {
public:
    HypreLibrary() {
        HYPRE_CALL(HYPRE_Initialize());
        HYPRE_CALL(HYPRE_SetMemoryLocation(location));
        HYPRE_CALL(HYPRE_SetExecutionPolicy(on_device ? HYPRE_EXEC_DEVICE : HYPRE_EXEC_HOST));
#ifdef HYPRE_USING_UMPIRE_DEVICE
        // Umpire reserves 4 GiB up front; OP_SOLVE_HYPRE_POOL (MiB) starts it smaller.
        if (const char *pool = std::getenv("OP_SOLVE_HYPRE_POOL"))
            HYPRE_CALL(HYPRE_SetUmpireDevicePoolSize((std::size_t)std::atol(pool) << 20));
#endif
    }

    ~HypreLibrary() override { HYPRE_Finalize(); }

    Order order() const override { return Order::diagonal_first; }

    bool direct() const override { return true; }

    std::unique_ptr<Matrix> matrix(op_mat A) override { return std::make_unique<HypreMatrix>(A); }

    std::unique_ptr<Solver> solver(op_solver s) override { return std::make_unique<HypreSolver>(s); }
};

} // namespace

std::unique_ptr<Library> make_hypre(int, char **) { return std::make_unique<HypreLibrary>(); }

} // namespace op::solve
