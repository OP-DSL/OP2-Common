// The Ginkgo backend: a distributed matrix whose local and non-local Csr
// parts are views of the op_mat's values, written in place by the loops (or,
// for blocks written by untranslated loops, a copy permuted from them),
// distributed vectors over views of the dats, and callback LinOps for the
// par-loop preconditioner and operator.

#include <op2/solve/solve.h>

#if OP2_SOLVE_DEVICE
#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#endif

#include <ginkgo/ginkgo.hpp>

#include <vector>

namespace op::solve {

namespace {

using vec = gko::matrix::Dense<double>;
using csr = gko::matrix::Csr<double, int>;
using dist_mtx = gko::experimental::distributed::Matrix<double, int, gko::int64>;
using dist_vec = gko::experimental::distributed::Vector<double>;
using partition = gko::experimental::distributed::Partition<int, gko::int64>;
using index_map = gko::experimental::distributed::index_map<int, gko::int64>;
using schwarz = gko::experimental::distributed::preconditioner::Schwarz<double, int, gko::int64>;
using jacobi = gko::preconditioner::Jacobi<double, int>;

std::shared_ptr<gko::Executor> make_executor() {
#if OP2_SOLVE_DEVICE
    int device;
    GPU_CHECK(gpuGetDevice(&device));
#if defined(OP2_HIP)
    return gko::HipExecutor::create(device, gko::OmpExecutor::create());
#else
    return gko::CudaExecutor::create(device, gko::OmpExecutor::create());
#endif
#else
    return gko::OmpExecutor::create();
#endif
}

std::shared_ptr<gko::Executor> executor;

template<typename T>
gko::array<T> to_exec(const std::vector<T>& host) {
    gko::array<T> on_host(executor->get_master(), host.begin(), host.end());
    return gko::array<T>(executor, on_host);
}

class GinkgoMatrix final : public Matrix {
public:
    explicit GinkgoMatrix(op_mat A) : m_A{A} {
        op_sparsity sp = A->sparsity;
        const int d = A->dim, n = sp->rows->size * d;
        gko::experimental::mpi::communicator comm{op::solve::comm()};

        std::vector<gko::int64> starts(comm.size() + 1, 0);
        gko::int64 start = sp->row_start * d;
        MPI_Allgather(&start, 1, MPI_LONG_LONG, starts.data(), 1, MPI_LONG_LONG, comm.get());
        starts[comm.size()] = sp->global_rows * d;
        auto ranges = to_exec(starts);
        auto parts = gko::share(partition::build_from_contiguous(executor, ranges));

        std::vector<gko::int64> remote(sp->garray.size() * d);
        for (std::size_t k = 0; k < sp->garray.size(); k++)
            for (int c = 0; c < d; c++)
                remote[k * d + c] = sp->garray[k] * d + c;
        auto recv = to_exec(remote);
        index_map imap(executor, parts, comm.rank(), recv);

        // In place, the values are the matrix's; blocked, a copy is.
        const Scalar& sc = A->scalar;
        const std::size_t nd = sc.diag.col.size(), no = sc.offd.col.size();
        double *data = (double *)memory(A->values);
        if (A->mode != Mode::direct) {
            m_staged = gko::array<double>(executor, nd + no);
            data = m_staged.get_data();
            std::vector<int> to((std::size_t)(sp->diag_blocks + sp->offd_blocks) * d * d);
            for (std::size_t k = 0; k < to.size(); k++)
                to[k] = position(sc, d, (int)(k / (d * d)), (int)(k % (d * d)));
            m_to = to_exec(to);
        }
        auto diag_values = gko::array<double>::view(executor, nd, data);
        auto offd_values = gko::array<double>::view(executor, no, data + nd);
        m_local = gko::share(csr::create(executor, gko::dim<2>(n, sp->cols->size * d), std::move(diag_values),
                                         to_exec(sc.diag.col), to_exec(sc.diag.ptr)));
        m_non_local = gko::share(csr::create(executor, gko::dim<2>(n, remote.size()), std::move(offd_values),
                                             to_exec(sc.offd.col), to_exec(sc.offd.ptr)));
        m_mat = gko::share(dist_mtx::create(executor, comm, std::move(imap), m_local, m_non_local));
    }

    void assemble() override {
        if (m_to.get_size() != 0)
            scatter(m_staged.get_data(), (const double *)memory(m_A->values), m_to.get_const_data(), m_to.get_size());
    }

    std::shared_ptr<dist_mtx> mat() const { return m_mat; }

private:
    op_mat m_A;
    std::shared_ptr<csr> m_local, m_non_local;
    std::shared_ptr<dist_mtx> m_mat;
    gko::array<double> m_staged;
    gko::array<int> m_to;
};

std::shared_ptr<dist_mtx> mat_of(op_mat A) { return static_cast<GinkgoMatrix *>(A->lib.get())->mat(); }

// A dat's owned values as a distributed vector.
std::unique_ptr<dist_vec> wrap(op_mat A, op_dat dat) {
    op_sparsity sp = A->sparsity;
    const gko::size_type n = (gko::size_type)sp->rows->size * A->dim, N = (gko::size_type)sp->global_rows * A->dim;
    auto local = vec::create(executor, gko::dim<2>(n, 1),
                             gko::array<double>::view(executor, n, (double *)memory(dat)), 1);
    return dist_vec::create(executor, gko::experimental::mpi::communicator{comm()}, gko::dim<2>(N, 1),
                            std::move(local));
}

// The par-loop preconditioner or operator, as a LinOp on distributed vectors.
class Callback : public gko::EnableLinOp<Callback>, public gko::EnableCreateMethod<Callback> {
public:
    Callback(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size = {}, op_solver s = nullptr,
             op_apply_fn fn = nullptr, void *ctx = nullptr)
        : gko::EnableLinOp<Callback>(exec, size), m_s{s}, m_fn{fn}, m_ctx{ctx} {}

protected:
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override {
        auto in = gko::as<dist_vec>(b);
        auto out = gko::as<dist_vec>(x);
        apply_through(m_s, m_fn, m_ctx, in->get_const_local_values(), out->get_local_values());
    }

    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b, const gko::LinOp *beta,
                    gko::LinOp *x) const override {
        auto out = gko::as<dist_vec>(x);
        auto y = gko::clone(out);
        apply_impl(b, y.get());
        out->scale(beta);
        out->add_scaled(alpha, y);
    }

private:
    op_solver m_s;
    op_apply_fn m_fn;
    void *m_ctx;
};

std::shared_ptr<gko::LinOpFactory> local_jacobi(int d, int n) {
    if (d == 1)
        return jacobi::build().with_max_block_size(1u).on(executor);
    std::vector<int> blocks(n / d + 1);
    for (std::size_t k = 0; k < blocks.size(); k++)
        blocks[k] = (int)k * d;
    return jacobi::build().with_max_block_size((gko::uint32)d).with_block_pointers(to_exec(blocks)).on(executor);
}

class GinkgoSolver final : public Solver {
public:
    explicit GinkgoSolver(op_solver s) : m_s{s} {}

    bool solve(op_dat b, op_dat x) override {
        try {
            build();
            auto bv = wrap(m_s->A, b), xv = wrap(m_s->A, x);
            auto logger = gko::share(gko::log::Convergence<double>::create());
            m_solver->add_logger(logger);
            m_solver->apply(bv, xv);
            m_solver->remove_logger(logger);

            auto norm = vec::create(executor->get_master(), gko::dim<2>(1, 1));
            bv->compute_norm2(norm);
            const double bnorm = norm->at(0, 0);
            auto rnorm = gko::make_temporary_clone(executor->get_master(), gko::as<vec>(logger->get_residual_norm()));
            iterations = (int)logger->get_num_iterations();
            residual = bnorm > 0 ? rnorm->at(0, 0) / bnorm : rnorm->at(0, 0);
            return logger->has_converged();
        } catch (const std::exception& e) {
            fail("Ginkgo: solver %s: %s", m_s->name.c_str(), e.what());
        }
    }

private:
    // The preconditioner and solver, made again once their matrices have been
    // assembled since.
    void build() {
        op_mat P = m_s->P != nullptr ? m_s->P : m_s->A;
        if (m_solver && m_state == m_s->A->state && m_pc_state == P->state)
            return;
        const Options& o = m_s->options;
        const int d = m_s->A->dim, n = m_s->A->sparsity->rows->size * d;
        const gko::size_type N = (gko::size_type)m_s->A->sparsity->global_rows * d;

        std::shared_ptr<const gko::LinOp> pc;
        auto Pm = mat_of(P);
        if (o.pc == "jacobi")
            pc = gko::share(schwarz::build().with_local_solver(local_jacobi(d, n)).on(executor)->generate(Pm));
        else if (o.pc == "ilu")
            pc = gko::share(schwarz::build()
                                .with_local_solver(gko::preconditioner::Ilu<>::build().with_factorization(
                                    gko::factorization::ParIlu<double, int>::build()))
                                .on(executor)
                                ->generate(Pm));
        else if (o.pc == "amg") {
            auto smoother = gko::share(gko::solver::build_smoother(
                gko::share(schwarz::build().with_local_solver(local_jacobi(1, n)).on(executor)), 2u, 0.9));
            // A stationary coarse solve, so the cycle is a fixed linear
            // operator, as CG needs of its preconditioner.
            auto coarsest = gko::share(gko::solver::build_smoother(
                gko::share(schwarz::build().with_local_solver(local_jacobi(1, n)).on(executor)), 4u, 0.9));
            pc = gko::share(gko::solver::Multigrid::build()
                                .with_mg_level(gko::multigrid::Pgm<double, int>::build().with_deterministic(true))
                                .with_pre_smoother(smoother)
                                .with_coarsest_solver(coarsest)
                                .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
                                .on(executor)
                                ->generate(Pm));
        } else if (o.pc == "custom")
            pc = gko::share(Callback::create(executor, gko::dim<2>(N, N), m_s, m_s->pc, m_s->pc_ctx));
        else if (o.pc != "none")
            fail("Ginkgo: unknown pc '%s'", o.pc.c_str());

        std::shared_ptr<const gko::LinOp> A = mat_of(m_s->A);
        if (m_s->op != nullptr)
            A = gko::share(Callback::create(executor, gko::dim<2>(N, N), m_s, m_s->op, m_s->op_ctx));

        auto iters = gko::stop::Iteration::build().with_max_iters((gko::size_type)o.maxit);
        auto norm = gko::stop::ResidualNorm<double>::build()
                        .with_baseline(gko::stop::mode::rhs_norm)
                        .with_reduction_factor(o.rtol);
        if (o.method == "cg")
            m_solver = gko::solver::Cg<double>::build()
                           .with_criteria(iters, norm)
                           .with_generated_preconditioner(pc)
                           .on(executor)
                           ->generate(A);
        else if (o.method == "gmres" || o.method == "fgmres")
            m_solver = gko::solver::Gmres<double>::build()
                           .with_criteria(iters, norm)
                           .with_krylov_dim((gko::size_type)o.restart)
                           .with_flexible(o.method == "fgmres")
                           .with_generated_preconditioner(pc)
                           .on(executor)
                           ->generate(A);
        else if (o.method == "bicgstab")
            m_solver = gko::solver::Bicgstab<double>::build()
                           .with_criteria(iters, norm)
                           .with_generated_preconditioner(pc)
                           .on(executor)
                           ->generate(A);
        else
            fail("Ginkgo: unknown method '%s'", o.method.c_str());
        m_state = m_s->A->state;
        m_pc_state = P->state;
    }

    op_solver m_s;
    std::shared_ptr<gko::LinOp> m_solver;
    int m_state = -1, m_pc_state = -1;
};

class GinkgoLibrary final : public Library {
public:
    GinkgoLibrary() { executor = make_executor(); }

    ~GinkgoLibrary() override { executor.reset(); }

    Order order() const override { return Order::sorted; }

    bool direct() const override { return true; }

    std::unique_ptr<Matrix> matrix(op_mat A) override { return std::make_unique<GinkgoMatrix>(A); }

    std::unique_ptr<Solver> solver(op_solver s) override { return std::make_unique<GinkgoSolver>(s); }
};

} // namespace

std::unique_ptr<Library> make_ginkgo(int, char **) { return std::make_unique<GinkgoLibrary>(); }

} // namespace op::solve
