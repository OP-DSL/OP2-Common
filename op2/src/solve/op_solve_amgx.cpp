// The AmgX backend: AmgX keeps its matrix in its own memory and takes values
// only by copy, so the loops write a region of OP2's laid out as AmgX takes
// them, which each assembly hands over (AMGX_matrix_replace_coefficients)
// before AmgX sets its solver up again.  A matrix's "format" is "bcsr", d x d
// blocks row major with global block columns (the default where AmgX has
// kernels for the block size), with an "external_diagonal" (the diagonal
// blocks apart, on one rank), or "csr", the scalar entries.  A solver is an
// AmgX configuration (config_version 2); the library's settings say whether
// to "print" AmgX's output and the AmgX configuration of its "resources".
// Device variants only: AmgX solves on the GPU.

#include <op2/solve/solve.h>

#include <amgx_c.h>

#if OP2_SOLVE_DEVICE
#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace op::solve {

namespace {

#define AMGX_CALL(call)                                                                              \
    do {                                                                                             \
        AMGX_RC rc_ = (call);                                                                        \
        if (rc_ != AMGX_RC_OK) {                                                                     \
            char message_[256];                                                                      \
            AMGX_get_error_string(rc_, message_, sizeof(message_));                                  \
            fail("amgx: %s at %s:%d", message_, __FILE__, __LINE__);                                 \
        }                                                                                            \
    } while (0)

constexpr AMGX_Mode mode = AMGX_mode_dDDI;

AMGX_resources_handle resources = nullptr;
MPI_Comm resources_comm;    // AmgX keeps a pointer to it
AMGX_config_handle base_config = nullptr;
bool verbose = false;

void print(const char *text, int) {
    if (verbose)
        std::fputs(text, stdout);
}

struct Format {
    bool scalar = false;
    bool diagonal = false;
};

// AmgX's own block kernels are for 1x1, 3x3 and 4x4 blocks; its general ones
// failed to launch at 6x6 on an H100, and the next run hung the GPU.
bool blocks_taken(int d) { return d == 1 || d == 3 || d == 4; }

Format parse_format(const Config& config, int d, const std::string& name) {
    Section c(config, "amgx: matrix " + name);
    const std::string format = c.get<std::string>("format", blocks_taken(d) ? "bcsr" : "csr");
    Format f;
    f.diagonal = c.get<bool>("external_diagonal", false);
    c.done();
    if (format != "bcsr" && format != "csr")
        fail("%s: format '%s', it has bcsr and csr", c.where().c_str(), format.c_str());
    f.scalar = format == "csr";
    if (f.diagonal && f.scalar)
        fail("%s: an external diagonal is bcsr's", c.where().c_str());
    return f;
}

class AmgxMatrix final : public Matrix {
public:
    AmgxMatrix(op_mat A, const Config& config) : m_A{A}, m_format{parse_format(config, A->dim, A->name)} {
        if (!on_device)
            fail("amgx: matrix %s: AmgX solves on the device, in a device variant", A->name.c_str());
        op_sparsity sp = A->sparsity;
        const int d = A->dim, nrows = sp->rows->size, bs = d * d;
        if (sp->rows != sp->cols)
            fail("amgx: matrix %s is not square", A->name.c_str());
        if (!m_format.scalar && !blocks_taken(d))
            fail("amgx: matrix %s: %d x %d blocks, which AmgX's block kernels do not take: use format csr",
                 A->name.c_str(), d, d);

        // Each row's blocks, the diagonal part's then the off-diagonal
        // part's, columns global: AmgX's upload order, which the layout
        // follows into the region.
        auto global = [&](int I, int k, bool offd) {
            return offd ? (std::int64_t)sp->garray[sp->offd_col[k]] : (std::int64_t)(sp->col_start + sp->diag_col[k]);
        };
        const std::size_t blocks = (std::size_t)sp->diag_blocks + sp->offd_blocks;
        if (m_format.scalar) {
            m_n = nrows * d;
            m_ptr.assign(m_n + 1, 0);
            std::size_t at = 0;
            for (int I = 0; I < nrows; I++) {
                const int nd = sp->diag_ptr[I + 1] - sp->diag_ptr[I], no = sp->offd_ptr[I + 1] - sp->offd_ptr[I];
                const int length = (nd + no) * d;
                for (int r = 0; r < d; r++) {
                    for (int part = 0; part < 2; part++) {
                        const auto& ptr = part == 0 ? sp->diag_ptr : sp->offd_ptr;
                        for (int k = ptr[I]; k < ptr[I + 1]; k++)
                            for (int c = 0; c < d; c++)
                                m_col.push_back(global(I, k, part == 1) * d + c);
                    }
                    m_ptr[I * d + r + 1] = m_ptr[I * d + r] + length;
                }
                int kb = 0;
                for (int k = sp->diag_ptr[I]; k < sp->diag_ptr[I + 1]; k++, kb++)
                    A->layout[k] = {(int)(at + (std::size_t)kb * d), length, 1, 0};
                for (int k = sp->offd_ptr[I]; k < sp->offd_ptr[I + 1]; k++, kb++)
                    A->layout[sp->diag_blocks + k] = {(int)(at + (std::size_t)kb * d), length, 1, 0};
                at += (std::size_t)length * d;
            }
            m_nnz = (int)m_col.size();
            lend(A, 0, (std::size_t)m_nnz * sizeof(double));
        } else {
            m_n = nrows;
            m_ptr.assign(m_n + 1, 0);
            const std::size_t diagonal_at = (blocks - (m_format.diagonal ? nrows : 0)) * bs;
            std::size_t next = 0;
            for (int I = 0; I < nrows; I++) {
                for (int part = 0; part < 2; part++) {
                    const auto& ptr = part == 0 ? sp->diag_ptr : sp->offd_ptr;
                    const int first = part == 0 ? 0 : sp->diag_blocks;
                    for (int k = ptr[I]; k < ptr[I + 1]; k++) {
                        if (m_format.diagonal && part == 0 && sp->diag_col[k] == I) {
                            A->layout[k] = {(int)(diagonal_at + (std::size_t)I * bs), d, 1, 0};
                            continue;
                        }
                        A->layout[first + k] = {(int)(next * bs), d, 1, 0};
                        m_col.push_back(global(I, k, part == 1));
                        next++;
                    }
                }
                m_ptr[I + 1] = (int)m_col.size();
            }
            m_nnz = (int)m_col.size();
            lend(A, 0, (blocks * bs) * sizeof(double));
            if (m_format.diagonal)
                m_diagonal = A->regions[0].data + diagonal_at * sizeof(double);
        }

        MPI_Comm c = comm();
        int ranks;
        MPI_Comm_size(c, &ranks);
        std::vector<std::int64_t> offsets(ranks + 1, 0);
        std::int64_t rows = m_n;
        MPI_Allgather(&rows, 1, MPI_INT64_T, offsets.data() + 1, 1, MPI_INT64_T, c);
        for (int r = 1; r <= ranks; r++)
            offsets[r] += offsets[r - 1];

        const int dim = m_format.scalar ? 1 : d;
        AMGX_CALL(AMGX_matrix_create(&m_matrix, resources, mode));
        // AmgX takes an external diagonal only on one GPU, columns local.
        if (m_format.diagonal) {
            if (ranks > 1)
                fail("amgx: matrix %s: bcsr:diag needs one rank: AmgX's distributed upload takes no external diagonal",
                     A->name.c_str());
            const std::vector<int> local(m_col.begin(), m_col.end());
            AMGX_CALL(AMGX_matrix_upload_all(m_matrix, m_n, m_nnz, dim, dim, m_ptr.data(), local.data(),
                                             A->regions[0].data, m_diagonal));
            return;
        }
        AMGX_distribution_handle distribution;
        AMGX_CALL(AMGX_distribution_create(&distribution, base_config));
        AMGX_CALL(AMGX_distribution_set_partition_data(distribution, AMGX_DIST_PARTITION_OFFSETS, offsets.data()));
        AMGX_CALL(AMGX_matrix_upload_distributed(m_matrix, (int)offsets[ranks], m_n, m_nnz, dim, dim, m_ptr.data(),
                                                 m_col.data(), A->regions[0].data, m_diagonal, distribution));
        AMGX_CALL(AMGX_distribution_destroy(distribution));
    }

    ~AmgxMatrix() override { AMGX_matrix_destroy(m_matrix); }

    void assemble() override {
        AMGX_CALL(AMGX_matrix_replace_coefficients(m_matrix, m_n, m_nnz, m_A->regions[0].data, m_diagonal));
    }

    AMGX_matrix_handle handle() const { return m_matrix; }
    int rows() const { return m_n; }
    int block() const { return m_format.scalar ? 1 : m_A->dim; }

private:
    op_mat m_A;
    Format m_format;
    AMGX_matrix_handle m_matrix = nullptr;
    int m_n = 0, m_nnz = 0;
    std::vector<int> m_ptr;
    std::vector<std::int64_t> m_col;
    const void *m_diagonal = nullptr;
};

AmgxMatrix& matrix(op_mat A) { return *static_cast<AmgxMatrix *>(A->lib.get()); }

// Without a configuration: GMRES and block Jacobi.
const Config default_settings = Config::parse(R"({"config_version": 2, "solver": {
    "solver": "GMRES", "max_iters": 1000, "tolerance": 1e-8, "convergence": "RELATIVE_INI_CORE",
    "norm": "L2", "gmres_n_restart": 30,
    "preconditioner": {"solver": "BLOCK_JACOBI", "max_iters": 1, "monitor_residual": 0}}})");

// The configuration as AmgX takes it, its outer solver keeping the residual
// history the layer reports from.
std::string configuration(op_solver s, const Config& config) {
    Config settings = config.is_null() ? default_settings : config;
    if (!settings.is_object() || !settings.contains("solver") || !settings["solver"].is_object())
        fail("amgx: solver %s: not an AmgX configuration with a solver", s->name.c_str());
    for (const char *key : {"monitor_residual", "store_res_history"})
        if (!settings["solver"].contains(key))
            settings["solver"][key] = 1;
    return settings.dump();
}

class AmgxSolver final : public Solver {
public:
    AmgxSolver(op_solver s, const Config& config) : m_s{s} {
        if (s->op != nullptr || s->pc != nullptr)
            fail("amgx: solver %s: AmgX takes no par-loop operator or preconditioner", s->name.c_str());
        const AmgxMatrix& A = matrix(s->A);
        if (s->P != nullptr && s->P != s->A)
            fail("amgx: solver %s: AmgX preconditions with its matrix", s->name.c_str());
        AMGX_CALL(AMGX_config_create(&m_config, configuration(s, config).c_str()));
        AMGX_CALL(AMGX_solver_create(&m_solver, resources, mode, m_config));
        AMGX_CALL(AMGX_vector_create(&m_b, resources, mode));
        AMGX_CALL(AMGX_vector_create(&m_x, resources, mode));
        AMGX_CALL(AMGX_vector_bind(m_b, A.handle()));
        AMGX_CALL(AMGX_vector_bind(m_x, A.handle()));
    }

    ~AmgxSolver() override {
        AMGX_solver_destroy(m_solver);
        AMGX_vector_destroy(m_b);
        AMGX_vector_destroy(m_x);
        AMGX_config_destroy(m_config);
    }

    bool solve(op_dat b, op_dat x) override {
        AmgxMatrix& A = matrix(m_s->A);
        if (m_state != m_s->A->state) {
            AMGX_CALL(AMGX_solver_setup(m_solver, A.handle()));
            m_state = m_s->A->state;
        }
        const int n = A.rows(), block = A.block();
        AMGX_CALL(AMGX_vector_upload(m_b, n, block, memory(b)));
        AMGX_CALL(AMGX_vector_upload(m_x, n, block, memory(x)));
        AMGX_CALL(AMGX_solver_solve(m_solver, m_b, m_x));
        AMGX_CALL(AMGX_vector_download(m_x, memory(x)));

        AMGX_SOLVE_STATUS status;
        AMGX_CALL(AMGX_solver_get_status(m_solver, &status));
        int its;
        AMGX_CALL(AMGX_solver_get_iterations_number(m_solver, &its));
        iterations = its;
        // Relative to ||b||, as the other backends report it.
        double rnorm = 0.0;
        AMGX_CALL(AMGX_solver_get_iteration_residual(m_solver, its, 0, &rnorm));
        const std::size_t count = (std::size_t)b->set->size * b->dim;
        std::vector<double> host(count);
        download(host.data(), memory(b), count * sizeof(double));
        double local = 0.0, bnorm;
        for (double v : host)
            local += v * v;
        MPI_Allreduce(&local, &bnorm, 1, MPI_DOUBLE, MPI_SUM, comm());
        bnorm = std::sqrt(bnorm);
        residual = bnorm > 0 ? rnorm / bnorm : rnorm;
        return status == AMGX_SOLVE_SUCCESS;
    }

private:
    op_solver m_s;
    AMGX_config_handle m_config = nullptr;
    AMGX_solver_handle m_solver = nullptr;
    AMGX_vector_handle m_b = nullptr, m_x = nullptr;
    int m_state = -1;
};

class AmgxLibrary final : public Library {
public:
    explicit AmgxLibrary(const Config& config) {
        Section c(config, "amgx: library");
        verbose = c.get<bool>("print", false);
        const Config resources_config =
            c.get<Config>("resources", Config::parse(R"({"config_version": 2, "solver": {"solver": "NOSOLVER"}})"));
        c.done();
        AMGX_CALL(AMGX_initialize());
        AMGX_CALL(AMGX_register_print_callback(print));
        AMGX_CALL(AMGX_config_create(&base_config, resources_config.dump().c_str()));
        resources_comm = comm();
        int device = 0;
#if OP2_SOLVE_DEVICE
        GPU_CHECK(gpuGetDevice(&device));
#endif
        AMGX_CALL(AMGX_resources_create(&resources, base_config, &resources_comm, 1, &device));
    }

    ~AmgxLibrary() override {
        AMGX_resources_destroy(resources);
        AMGX_config_destroy(base_config);
        AMGX_finalize();
    }

    std::unique_ptr<Matrix> matrix(op_mat A, const Config& config) override {
        return std::make_unique<AmgxMatrix>(A, config);
    }

    std::unique_ptr<Solver> solver(op_solver s, const Config& config) override {
        return std::make_unique<AmgxSolver>(s, config);
    }
};

} // namespace

std::unique_ptr<Library> make_amgx(const Config& config) { return std::make_unique<AmgxLibrary>(config); }

} // namespace op::solve
