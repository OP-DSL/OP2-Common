// The hypre backend: a ParCSR matrix whose diagonal and off-diagonal parts'
// data are the op_mat's values, written in place by the loops ("format":
// "csr", each row's diagonal entry first), Krylov solvers over ParVectors
// wrapping the dats.  Settings are hypre's own setters by name, without
// their prefix: a solver's "solver" ("PCG", "GMRES", "FlexGMRES",
// "BiCGSTAB") and its setters ("Tol", "MaxIter", "KDim", ...), "precond"
// ("BoomerAMG", "ILU", "DiagScale") and an object of its setters under its
// name; the library's global ones ("UmpireDevicePoolSize", "SpMVUseVendor",
// ...).

#include <op2/solve/solve.h>

#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <_hypre_parcsr_mv.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <span>
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
    HypreMatrix(op_mat A, const Config& config) : m_A{A} {
        Section c(config, "hypre: matrix " + A->name);
        const std::string format = c.get<std::string>("format", "csr");
        c.done();
        if (format != "csr")
            fail("%s: format '%s', hypre keeps CSR", c.where().c_str(), format.c_str());
        op_sparsity sp = A->sparsity;
        const int d = A->dim, nrows = sp->rows->size;
        const idx_g_t global = std::max(sp->global_rows, sp->global_cols) * d;
        if (global > std::numeric_limits<HYPRE_BigInt>::max())
            fail("hypre: matrix %s has %lld global rows or columns, past its %zu-byte HYPRE_BigInt (build hypre with "
                 "HYPRE_ENABLE_MIXEDINT)", A->name.c_str(), (long long)global, sizeof(HYPRE_BigInt));
        HYPRE_BigInt rows[2] = {(HYPRE_BigInt)(sp->row_start * d), (HYPRE_BigInt)((sp->row_start + nrows) * d)};
        HYPRE_BigInt cols[2] = {(HYPRE_BigInt)(sp->col_start * d),
                                (HYPRE_BigInt)((sp->col_start + sp->cols->size) * d)};

        // The diagonal part's values, each row's diagonal entry first, then
        // the off-diagonal part's, in one region that is hypre's data.
        const std::size_t nd = (std::size_t)part_blocks(A, Part::diag) * d * d;
        const std::size_t no = (std::size_t)part_blocks(A, Part::offd) * d * d;
        double *data = (double *)lend(A, 0, (nd + no) * sizeof(double));
        const ScalarCsr dsc = csr_layout(A, Part::diag, 0, 0, Diagonal::first);
        const ScalarCsr osc = csr_layout(A, Part::offd, 0, (int)nd);

        std::vector<HYPRE_Int> diag_i(dsc.ptr.begin(), dsc.ptr.end()), diag_j(dsc.col.begin(), dsc.col.end());
        std::vector<HYPRE_Int> offd_i(osc.ptr.begin(), osc.ptr.end()), offd_j(osc.col.begin(), osc.col.end());
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
        hypre_CSRMatrixData(diag) = data;
        hypre_CSRMatrixData(offd) = data + nd;
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
    }

    // The loops wrote hypre's data.
    void assemble() override {}

    hypre_ParCSRMatrix *par() const { return m_par; }

private:
    op_mat m_A;
    hypre_ParCSRMatrix *m_par = nullptr;
    HYPRE_Int *m_diag_j = nullptr, *m_offd_j = nullptr;
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

// A setter by its name in hypre (HYPRE_<prefix>Set<name>), of an integer or
// a real.
struct Setter {
    const char *name;
    HYPRE_Int (*integer)(HYPRE_Solver, HYPRE_Int);
    HYPRE_Int (*real)(HYPRE_Solver, HYPRE_Real);
};
#define HYPRE_INT(prefix, name) {#name, prefix##Set##name, nullptr}
#define HYPRE_REAL(prefix, name) {#name, nullptr, prefix##Set##name}

// An object's members by the setters of its table, but for those skipped.
void set(HYPRE_Solver solver, Section& c, std::span<const Setter> setters, std::span<const char *const> skip = {}) {
    for (const auto& [key, value] : c.config().items()) {
        if (std::find_if(skip.begin(), skip.end(), [&](const char *k) { return key == k; }) != skip.end())
            continue;
        auto setter = std::find_if(setters.begin(), setters.end(), [&](const Setter& s) { return key == s.name; });
        if (setter == setters.end())
            fail("%s: unknown member '%s'", c.where().c_str(), key.c_str());
        if (setter->integer != nullptr)
            HYPRE_CALL(setter->integer(solver, (HYPRE_Int)c.get<long>(key.c_str())));
        else
            HYPRE_CALL(setter->real(solver, (HYPRE_Real)c.get<double>(key.c_str())));
    }
}

const Setter boomeramg_setters[] = {
    HYPRE_INT(HYPRE_BoomerAMG, CoarsenType),     HYPRE_INT(HYPRE_BoomerAMG, InterpType),
    HYPRE_INT(HYPRE_BoomerAMG, RelaxType),       HYPRE_INT(HYPRE_BoomerAMG, RelaxOrder),
    HYPRE_INT(HYPRE_BoomerAMG, NumSweeps),       HYPRE_INT(HYPRE_BoomerAMG, MaxLevels),
    HYPRE_INT(HYPRE_BoomerAMG, MaxCoarseSize),   HYPRE_INT(HYPRE_BoomerAMG, MinCoarseSize),
    HYPRE_REAL(HYPRE_BoomerAMG, StrongThreshold), HYPRE_REAL(HYPRE_BoomerAMG, MaxRowSum),
    HYPRE_REAL(HYPRE_BoomerAMG, TruncFactor),    HYPRE_INT(HYPRE_BoomerAMG, PMaxElmts),
    HYPRE_INT(HYPRE_BoomerAMG, AggNumLevels),    HYPRE_INT(HYPRE_BoomerAMG, AggInterpType),
    HYPRE_INT(HYPRE_BoomerAMG, AggPMaxElmts),    HYPRE_INT(HYPRE_BoomerAMG, NumPaths),
    HYPRE_INT(HYPRE_BoomerAMG, NumFunctions),    HYPRE_INT(HYPRE_BoomerAMG, Nodal),
    HYPRE_INT(HYPRE_BoomerAMG, NodalDiag),       HYPRE_INT(HYPRE_BoomerAMG, KeepTranspose),
    HYPRE_INT(HYPRE_BoomerAMG, RAP2),            HYPRE_INT(HYPRE_BoomerAMG, ModuleRAP2),
    HYPRE_INT(HYPRE_BoomerAMG, CycleType),       HYPRE_REAL(HYPRE_BoomerAMG, Tol),
    HYPRE_INT(HYPRE_BoomerAMG, MaxIter),         HYPRE_INT(HYPRE_BoomerAMG, PrintLevel),
    HYPRE_INT(HYPRE_BoomerAMG, Logging),         HYPRE_REAL(HYPRE_BoomerAMG, RelaxWt),
    HYPRE_REAL(HYPRE_BoomerAMG, OuterWt),        HYPRE_INT(HYPRE_BoomerAMG, SmoothType),
    HYPRE_INT(HYPRE_BoomerAMG, SmoothNumLevels), HYPRE_INT(HYPRE_BoomerAMG, SmoothNumSweeps),
};

const Setter ilu_setters[] = {
    HYPRE_INT(HYPRE_ILU, Type),          HYPRE_INT(HYPRE_ILU, LevelOfFill),      HYPRE_INT(HYPRE_ILU, MaxIter),
    HYPRE_REAL(HYPRE_ILU, Tol),          HYPRE_REAL(HYPRE_ILU, DropThreshold),   HYPRE_INT(HYPRE_ILU, MaxNnzPerRow),
    HYPRE_INT(HYPRE_ILU, LocalReordering), HYPRE_INT(HYPRE_ILU, TriSolve),       HYPRE_INT(HYPRE_ILU, LowerJacobiIters),
    HYPRE_INT(HYPRE_ILU, UpperJacobiIters), HYPRE_INT(HYPRE_ILU, PrintLevel),    HYPRE_INT(HYPRE_ILU, Logging),
};

// A Krylov solver's entry points and setters.
struct Krylov {
    const char *name;
    HYPRE_Int (*create)(MPI_Comm, HYPRE_Solver *);
    HYPRE_Int (*destroy)(HYPRE_Solver);
    HYPRE_Int (*setup)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
    HYPRE_Int (*solve)(HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector);
    HYPRE_Int (*precond)(HYPRE_Solver, HYPRE_PtrToParSolverFcn, HYPRE_PtrToParSolverFcn, HYPRE_Solver);
    HYPRE_Int (*iterations)(HYPRE_Solver, HYPRE_Int *);
    HYPRE_Int (*residual)(HYPRE_Solver, HYPRE_Real *);
    std::vector<Setter> setters;
};

#define HYPRE_KRYLOV(name)                                                                               \
    #name, HYPRE_ParCSR##name##Create, HYPRE_ParCSR##name##Destroy, HYPRE_ParCSR##name##Setup,          \
        HYPRE_ParCSR##name##Solve, HYPRE_ParCSR##name##SetPrecond, HYPRE_ParCSR##name##GetNumIterations, \
        HYPRE_ParCSR##name##GetFinalRelativeResidualNorm

const Krylov krylovs[] = {
    {HYPRE_KRYLOV(PCG),
     {HYPRE_REAL(HYPRE_ParCSRPCG, Tol), HYPRE_REAL(HYPRE_ParCSRPCG, AbsoluteTol), HYPRE_INT(HYPRE_ParCSRPCG, MaxIter),
      HYPRE_INT(HYPRE_ParCSRPCG, TwoNorm), HYPRE_INT(HYPRE_ParCSRPCG, RelChange),
      HYPRE_INT(HYPRE_ParCSRPCG, PrintLevel), HYPRE_INT(HYPRE_ParCSRPCG, Logging)}},
    {HYPRE_KRYLOV(GMRES),
     {HYPRE_REAL(HYPRE_ParCSRGMRES, Tol), HYPRE_REAL(HYPRE_ParCSRGMRES, AbsoluteTol),
      HYPRE_INT(HYPRE_ParCSRGMRES, MaxIter), HYPRE_INT(HYPRE_ParCSRGMRES, MinIter), HYPRE_INT(HYPRE_ParCSRGMRES, KDim),
      HYPRE_INT(HYPRE_ParCSRGMRES, PrintLevel), HYPRE_INT(HYPRE_ParCSRGMRES, Logging)}},
    {HYPRE_KRYLOV(FlexGMRES),
     {HYPRE_REAL(HYPRE_ParCSRFlexGMRES, Tol), HYPRE_REAL(HYPRE_ParCSRFlexGMRES, AbsoluteTol),
      HYPRE_INT(HYPRE_ParCSRFlexGMRES, MaxIter), HYPRE_INT(HYPRE_ParCSRFlexGMRES, MinIter),
      HYPRE_INT(HYPRE_ParCSRFlexGMRES, KDim), HYPRE_INT(HYPRE_ParCSRFlexGMRES, PrintLevel),
      HYPRE_INT(HYPRE_ParCSRFlexGMRES, Logging)}},
    {HYPRE_KRYLOV(BiCGSTAB),
     {HYPRE_REAL(HYPRE_ParCSRBiCGSTAB, Tol), HYPRE_REAL(HYPRE_ParCSRBiCGSTAB, AbsoluteTol),
      HYPRE_INT(HYPRE_ParCSRBiCGSTAB, MaxIter), HYPRE_INT(HYPRE_ParCSRBiCGSTAB, MinIter),
      HYPRE_INT(HYPRE_ParCSRBiCGSTAB, PrintLevel), HYPRE_INT(HYPRE_ParCSRBiCGSTAB, Logging)}},
};

// Without a configuration: GMRES and diagonal scaling, hypre's defaults else.
const Config default_settings = {{"solver", "GMRES"}, {"precond", "DiagScale"}};

class HypreSolver final : public Solver {
public:
    HypreSolver(op_solver s, const Config& config) : m_s{s} {
        if (s->op != nullptr)
            fail("hypre: solver %s: a par-loop operator is not supported", s->name.c_str());
        Config settings = config.is_null() ? default_settings : config;
        if (config.is_null() && s->pc != nullptr)
            settings.erase("precond");
        Section c(settings, "hypre: solver " + s->name);

        const std::string name = c.get<std::string>("solver", "GMRES");
        auto found = std::find_if(std::begin(krylovs), std::end(krylovs),
                                  [&](const Krylov& k) { return name == k.name; });
        if (found == std::end(krylovs))
            fail("%s: solver '%s', hypre has PCG, GMRES, FlexGMRES and BiCGSTAB", c.where().c_str(), name.c_str());
        m_krylov = &*found;
        HYPRE_CALL(m_krylov->create(comm(), &m_solver));
        if (name == "PCG")
            HYPRE_CALL(HYPRE_ParCSRPCGSetTwoNorm(m_solver, 1));

        if (s->pc != nullptr && c.has("precond"))
            fail("%s: names a precond, and has a par-loop preconditioner", c.where().c_str());
        const std::string precond = c.get<std::string>("precond", "");
        static const char *const own[] = {"solver", "precond", "BoomerAMG", "ILU"};
        set(m_solver, c, m_krylov->setters, own);
        if (precond == "BoomerAMG") {
            // One V-cycle, systems AMG for blocks, coarsening and smoothing the
            // device takes; then the configuration's.
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
        } else if (precond == "ILU") {
            HYPRE_CALL(HYPRE_ILUCreate(&m_pc));
            HYPRE_CALL(HYPRE_ILUSetMaxIter(m_pc, 1));
            HYPRE_CALL(HYPRE_ILUSetTol(m_pc, 0.0));
            m_apply = HYPRE_ILUSolve;
            m_setup = HYPRE_ILUSetup;
            m_pc_destroy = HYPRE_ILUDestroy;
        } else if (precond == "DiagScale") {
            m_apply = HYPRE_ParCSRDiagScale;
            m_setup = HYPRE_ParCSRDiagScaleSetup;
        } else if (!precond.empty())
            fail("%s: precond '%s', hypre has BoomerAMG, ILU and DiagScale", c.where().c_str(), precond.c_str());
        for (const char *sub : {"BoomerAMG", "ILU"})
            if (c.has(sub)) {
                if (precond != sub)
                    fail("%s: settings of %s, which is not its precond", c.where().c_str(), sub);
                Section pc(c.config()[sub], c.where() + ": " + sub);
                set(m_pc, pc, sub == std::string("BoomerAMG") ? std::span<const Setter>(boomeramg_setters)
                                                               : std::span<const Setter>(ilu_setters));
            }
        if (s->pc != nullptr) {
            m_apply = custom_apply;
            m_pc = (HYPRE_Solver)s;
        }
        if (m_apply != nullptr)
            HYPRE_CALL(m_krylov->precond(m_solver, m_apply, no_setup, m_pc));

        m_b = wrap(s->A);
        m_x = wrap(s->A);
    }

    ~HypreSolver() override {
        m_krylov->destroy(m_solver);
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
            HYPRE_CALL(m_krylov->setup(m_solver, A, bv, xv));
            m_ready = true;
        }
        if (m_setup != nullptr && m_pc_state != P->state) {
            HYPRE_CALL(m_setup(m_pc, (HYPRE_ParCSRMatrix)par_of(P), bv, xv));
            m_pc_state = P->state;
        }
        // Stopping short of the tolerance is an error hypre flags, not a failure.
        const HYPRE_Int error = m_krylov->solve(m_solver, A, bv, xv);
        if (error & ~HYPRE_ERROR_CONV)
            fail("hypre: solver %s: error %d", m_s->name.c_str(), (int)error);
        HYPRE_ClearAllErrors();

        HYPRE_Int its;
        HYPRE_Real norm;
        HYPRE_CALL(m_krylov->iterations(m_solver, &its));
        HYPRE_CALL(m_krylov->residual(m_solver, &norm));
        iterations = (int)its;
        residual = (double)norm;
        return !(error & HYPRE_ERROR_CONV);
    }

private:
    op_solver m_s;
    const Krylov *m_krylov = nullptr;
    HYPRE_Solver m_solver = nullptr, m_pc = nullptr;
    HYPRE_PtrToParSolverFcn m_apply = nullptr, m_setup = nullptr;
    HYPRE_Int (*m_pc_destroy)(HYPRE_Solver) = nullptr;
    hypre_ParVector *m_b = nullptr, *m_x = nullptr;
    bool m_ready = false;
    int m_pc_state = -1;
};

// hypre's global settings, which take sizes and switches.
struct Global {
    const char *name;
    HYPRE_Int (*size)(std::size_t);
    HYPRE_Int (*integer)(HYPRE_Int);
};

const Global globals[] = {
#ifdef HYPRE_USING_UMPIRE
    {"UmpireDevicePoolSize", HYPRE_SetUmpireDevicePoolSize, nullptr},
    {"UmpireUMPoolSize", HYPRE_SetUmpireUMPoolSize, nullptr},
    {"UmpireHostPoolSize", HYPRE_SetUmpireHostPoolSize, nullptr},
    {"UmpirePinnedPoolSize", HYPRE_SetUmpirePinnedPoolSize, nullptr},
#endif
    {"SpMVUseVendor", nullptr, HYPRE_SetSpMVUseVendor},
    {"SpGemmUseVendor", nullptr, HYPRE_SetSpGemmUseVendor},
    {"SpTransUseVendor", nullptr, HYPRE_SetSpTransUseVendor},
    {"UseGpuRand", nullptr, HYPRE_SetUseGpuRand},
    {"LogLevel", nullptr, HYPRE_SetLogLevel},
};

class HypreLibrary final : public Library {
public:
    explicit HypreLibrary(const Config& config) {
        HYPRE_CALL(HYPRE_Initialize());
        HYPRE_CALL(HYPRE_SetMemoryLocation(location));
        HYPRE_CALL(HYPRE_SetExecutionPolicy(on_device ? HYPRE_EXEC_DEVICE : HYPRE_EXEC_HOST));
        Section c(config, "hypre: library");
        if (config.is_object())
            for (const auto& [key, value] : config.items()) {
                auto global = std::find_if(std::begin(globals), std::end(globals),
                                           [&](const Global& g) { return key == g.name; });
                if (global == std::end(globals))
                    fail("%s: unknown member '%s' (or not in this build)", c.where().c_str(), key.c_str());
                if (global->size != nullptr)
                    HYPRE_CALL(global->size((std::size_t)c.get<long>(key.c_str())));
                else
                    HYPRE_CALL(global->integer((HYPRE_Int)c.get<long>(key.c_str())));
            }
    }

    ~HypreLibrary() override { HYPRE_Finalize(); }

    std::unique_ptr<Matrix> matrix(op_mat A, const Config& config) override {
        return std::make_unique<HypreMatrix>(A, config);
    }

    std::unique_ptr<Solver> solver(op_solver s, const Config& config) override {
        return std::make_unique<HypreSolver>(s, config);
    }
};

} // namespace

std::unique_ptr<Library> make_hypre(const Config& config) { return std::make_unique<HypreLibrary>(config); }

} // namespace op::solve
