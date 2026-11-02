// The Ginkgo backend: a distributed matrix whose local and non-local parts
// are views of the op_mat's values, written in place by the loops ("format":
// "csr", with a "strategy" "cusparse", "classical" or "merge_path", "coo",
// "ell", "fbcsr"), or Ginkgo's own arrays borrowed ("sellp"); distributed
// vectors over views of the dats; and callback LinOps for the par-loop
// preconditioner and operator.  A solver is a Ginkgo configuration (its file
// configuration, gko::config), which finds a preconditioner built apart from
// it (from P, or the par-loop one) as "generated_preconditioner", and Jacobi
// in blocks of a node's components, which Ginkgo's configuration cannot
// give it, as the factory "op2_block_jacobi".

#include <op2/solve/solve.h>

#if OP2_SOLVE_DEVICE
#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#endif

#include <ginkgo/ginkgo.hpp>

#include <algorithm>
#include <vector>

namespace op::solve {

namespace {

using vec = gko::matrix::Dense<double>;
using csr = gko::matrix::Csr<double, int>;
using coo = gko::matrix::Coo<double, int>;
using ell = gko::matrix::Ell<double, int>;
using sellp = gko::matrix::Sellp<double, int>;
using fbcsr = gko::matrix::Fbcsr<double, int>;
using dist_mtx = gko::experimental::distributed::Matrix<double, int, gko::int64>;
using dist_vec = gko::experimental::distributed::Vector<double>;
using partition = gko::experimental::distributed::Partition<int, gko::int64>;
using index_map = gko::experimental::distributed::index_map<int, gko::int64>;
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
    GinkgoMatrix(op_mat A, const Config& config) : m_A{A} {
        Section c(config, "ginkgo: matrix " + A->name);
        std::string kind = c.get<std::string>("format", "csr");
        const std::string strategy = c.get<std::string>("strategy", "");
        c.done();
        if (!strategy.empty() && kind != "csr")
            fail("%s: a strategy is for csr", c.where().c_str());
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

        // Each part in the format asked for: views of one region, the local
        // part's values then the non-local part's (csr, coo, ell, fbcsr), or
        // Ginkgo's own arrays, borrowed (sellp).  1 x 1 blocks are scalar
        // entries, which Csr holds (and Fbcsr's diagonal is not extracted on
        // the device).
        if (kind == "fbcsr" && d == 1)
            kind = "csr";
        const gko::dim<2> sizes[2] = {gko::dim<2>(n, sp->cols->size * d), gko::dim<2>(n, remote.size())};
        const Part which[2] = {Part::diag, Part::offd};
        std::shared_ptr<gko::LinOp> made[2];
        if (kind == "sellp") {
            // Slices of whole blocks, near Ginkgo's default of 64 rows.
            const int slice = std::max(d, 64 / d * d);
            for (int p = 0; p < 2; p++) {
                const Sell sell = sell_layout(A, which[p], p, 0, slice, 1, -1);
                auto m = gko::share(sellp::create(executor, sizes[p], sell.slice, 1, sell.sets.back()));
                const std::vector<gko::size_type> lengths(sell.lengths.begin(), sell.lengths.end()),
                    sets(sell.sets.begin(), sell.sets.end());
                executor->copy_from(executor->get_master(), lengths.size(), lengths.data(), m->get_slice_lengths());
                executor->copy_from(executor->get_master(), sets.size(), sets.data(), m->get_slice_sets());
                executor->copy_from(executor->get_master(), sell.col.size(), sell.col.data(), m->get_col_idxs());
                borrow(A, p, m->get_values(), sell.col.size() * sizeof(double));
                made[p] = m;
            }
        } else {
            std::size_t counts[2];
            ScalarCsr scalar[2];
            BlockCsr blocks[2];
            Ell padded[2];
            for (int p = 0; p < 2; p++) {
                const int offset = p == 0 ? 0 : (int)counts[0];
                if (kind == "csr" || kind == "coo") {
                    scalar[p] = csr_layout(A, which[p], 0, offset);
                    counts[p] = scalar[p].col.size();
                } else if (kind == "fbcsr") {
                    blocks[p] = bcsr_layout(A, which[p], 0, offset, Major::column);
                    counts[p] = blocks[p].col.size() * d * d;
                } else if (kind == "ell") {
                    padded[p] = ell_layout(A, which[p], 0, offset, -1);
                    counts[p] = padded[p].col.size();
                } else
                    fail("%s: format '%s', it has csr, coo, ell, sellp and fbcsr", c.where().c_str(), kind.c_str());
            }
            double *data = (double *)lend(A, 0, (counts[0] + counts[1]) * sizeof(double));
            for (int p = 0; p < 2; p++) {
                auto values = gko::array<double>::view(executor, counts[p], data + (p == 0 ? 0 : counts[0]));
                if (kind == "csr") {
                    auto m = csr::create(executor, sizes[p], std::move(values), to_exec(scalar[p].col),
                                         to_exec(scalar[p].ptr));
                    if (strategy == "cusparse")
                        m->set_strategy(std::make_shared<csr::sparselib>());
                    else if (strategy == "classical")
                        m->set_strategy(std::make_shared<csr::classical>());
                    else if (strategy == "merge_path")
                        m->set_strategy(std::make_shared<csr::merge_path>());
                    else if (!strategy.empty())
                        fail("%s: strategy '%s', it has cusparse, classical and merge_path", c.where().c_str(),
                             strategy.c_str());
                    made[p] = gko::share(std::move(m));
                } else if (kind == "coo") {
                    std::vector<int> rows;
                    for (std::size_t i = 0; i + 1 < scalar[p].ptr.size(); i++)
                        rows.insert(rows.end(), scalar[p].ptr[i + 1] - scalar[p].ptr[i], (int)i);
                    made[p] = gko::share(
                        coo::create(executor, sizes[p], std::move(values), to_exec(scalar[p].col), to_exec(rows)));
                } else if (kind == "fbcsr") {
                    made[p] = gko::share(fbcsr::create(executor, sizes[p], d, std::move(values),
                                                       to_exec(blocks[p].col), to_exec(blocks[p].ptr)));
                } else {
                    made[p] = gko::share(ell::create(executor, sizes[p], std::move(values), to_exec(padded[p].col),
                                                     (gko::size_type)padded[p].width, (gko::size_type)n));
                }
            }
        }
        m_local = made[0];
        m_non_local = made[1];
        m_mat = gko::share(dist_mtx::create(executor, comm, std::move(imap), m_local, m_non_local));
    }

    // The loops wrote the parts' values.
    void assemble() override {}

    std::shared_ptr<dist_mtx> mat() const { return m_mat; }

private:
    op_mat m_A;
    std::shared_ptr<gko::LinOp> m_local, m_non_local;
    std::shared_ptr<dist_mtx> m_mat;
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

// A configuration as Ginkgo's property tree.
gko::config::pnode tree(const Config& c) {
    if (c.is_object()) {
        gko::config::pnode::map_type members;
        for (const auto& [key, value] : c.items())
            members.emplace(key, tree(value));
        return gko::config::pnode(members);
    }
    if (c.is_array()) {
        gko::config::pnode::array_type items;
        for (const Config& value : c)
            items.push_back(tree(value));
        return gko::config::pnode(items);
    }
    if (c.is_boolean())
        return gko::config::pnode(c.get<bool>());
    if (c.is_number_integer())
        return gko::config::pnode(c.get<long long>());
    if (c.is_number_float())
        return gko::config::pnode(c.get<double>());
    if (c.is_string())
        return gko::config::pnode(c.get<std::string>());
    return gko::config::pnode();
}

// Without a configuration: GMRES, Schwarz with block Jacobi on each rank's
// part.
const Config default_settings = Config::parse(R"({
    "type": "solver::Gmres", "krylov_dim": 30,
    "criteria": {"iteration": 1000, "relative_residual_norm": 1e-8},
    "preconditioner": {"type": "preconditioner::Schwarz", "local_solver": "op2_block_jacobi"}})");

class GinkgoSolver final : public Solver {
public:
    GinkgoSolver(op_solver s, const Config& config) : m_s{s}, m_settings(config.is_null() ? default_settings : config) {
        if (!m_settings.is_object())
            fail("ginkgo: solver %s: not a JSON object", s->name.c_str());
        if (s->pc != nullptr && m_settings.contains("preconditioner")) {
            if (!config.is_null())
                fail("ginkgo: solver %s: names a preconditioner, and has a par-loop one", s->name.c_str());
            m_settings.erase("preconditioner");
        }
    }

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
            fail("ginkgo: solver %s: %s", m_s->name.c_str(), e.what());
        }
    }

private:
    // The solver, made again once its matrices have been assembled since.  A
    // preconditioner from P, or from A under a par-loop operator, or the
    // par-loop one, is made apart and found in the registry.
    void build() {
        op_mat P = m_s->P != nullptr ? m_s->P : m_s->A;
        if (m_solver && m_state == m_s->A->state && m_pc_state == P->state)
            return;
        const gko::size_type N = (gko::size_type)m_s->A->sparsity->global_rows * m_s->A->dim;
        Config settings = m_settings;
        gko::config::registry registry;
        const int d = m_s->A->dim, n = m_s->A->sparsity->rows->size * d;
        std::vector<int> blocks(n / d + 1);
        for (std::size_t k = 0; k < blocks.size(); k++)
            blocks[k] = (int)k * d;
        registry.emplace("op2_block_jacobi", gko::share(jacobi::build()
                                                            .with_max_block_size((gko::uint32)d)
                                                            .with_block_pointers(to_exec(blocks))
                                                            .with_skip_sorting(true)
                                                            .on(executor)));
        std::shared_ptr<const gko::LinOp> A = mat_of(m_s->A);
        if (m_s->op != nullptr)
            A = gko::share(Callback::create(executor, gko::dim<2>(N, N), m_s, m_s->op, m_s->op_ctx));
        std::shared_ptr<gko::LinOp> pc;
        if (m_s->pc != nullptr)
            pc = gko::share(Callback::create(executor, gko::dim<2>(N, N), m_s, m_s->pc, m_s->pc_ctx));
        else if ((m_s->P != nullptr || m_s->op != nullptr) && settings.contains("preconditioner")) {
            pc = gko::share(
                gko::config::parse(tree(settings["preconditioner"]), registry).on(executor)->generate(mat_of(P)));
            settings.erase("preconditioner");
        }
        if (pc) {
            registry.emplace("op2_preconditioner", pc);
            settings["generated_preconditioner"] = "op2_preconditioner";
        }
        m_solver = gko::share(gko::config::parse(tree(settings), registry).on(executor)->generate(A));
        m_state = m_s->A->state;
        m_pc_state = P->state;
    }

    op_solver m_s;
    Config m_settings;
    std::shared_ptr<gko::LinOp> m_solver;
    int m_state = -1, m_pc_state = -1;
};

class GinkgoLibrary final : public Library {
public:
    explicit GinkgoLibrary(const Config& config) {
        Section(config, "ginkgo: library").done();
        executor = make_executor();
    }

    ~GinkgoLibrary() override { executor.reset(); }

    std::unique_ptr<Matrix> matrix(op_mat A, const Config& config) override {
        return std::make_unique<GinkgoMatrix>(A, config);
    }

    std::unique_ptr<Solver> solver(op_solver s, const Config& config) override {
        return std::make_unique<GinkgoSolver>(s, config);
    }
};

} // namespace

std::unique_ptr<Library> make_ginkgo(const Config& config) { return std::make_unique<GinkgoLibrary>(config); }

} // namespace op::solve
