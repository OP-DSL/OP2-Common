// The mock library: a stand-in for a solver library that allocates and owns
// its matrix in its own layout, its local and non-local parts apart, as
// SPLISS does.  Formats: "compact" (block CSR, as CompactBlockStorage) and
// "bell" (block ELL, as BELLStorage), each with ":column" for column-major
// blocks, and "bell:interleaved" for each block entry interleaved across the
// block rows, coalesced for a thread per row.  Loops write its storage in
// place; it solves with CG (point Jacobi, none, or a par-loop
// preconditioner), its non-local columns by its own exchange.

#include <op2/solve/solve.h>

#include "mock.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

namespace op::solve {

namespace mock {

#if !OP2_SOLVE_DEVICE
double *allocate(std::size_t count) { return (double *)std::calloc(count + 1, sizeof(double)); }
void free(void *memory) { std::free(memory); }
void to_host(void *dst, const void *src, std::size_t bytes) { std::memcpy(dst, src, bytes); }
void from_host(void *dst, const void *src, std::size_t bytes) { std::memcpy(dst, src, bytes); }

void spmv(const Format& f, int d, int n, const Part& local, const Part& nonlocal, const double *x, const double *halo,
          double *y) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        y[i] = row_product(f, d, local, i, x) + (nonlocal.rows > 0 ? row_product(f, d, nonlocal, i, halo) : 0.0);
}

void inverse_diagonal(const Format& f, int d, int n, const Part& local, double *inv) {
    for (int i = 0; i < n; i++)
        inv[i] = 1.0 / diagonal(f, d, local, i);
}

double dot(int n, const double *a, const double *b) {
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

void axpy(int n, double alpha, const double *x, double *y) {
    for (int i = 0; i < n; i++)
        y[i] += alpha * x[i];
}

void xpay(int n, const double *x, double beta, double *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + beta * y[i];
}

void multiply(int n, const double *a, const double *b, double *y) {
    for (int i = 0; i < n; i++)
        y[i] = a[i] * b[i];
}
#endif

} // namespace mock

namespace {

mock::Format parse_format(const std::string& text, const std::string& name) {
    mock::Format f;
    std::istringstream words(text.empty() ? "compact" : text);
    std::string word;
    bool first = true;
    while (std::getline(words, word, ':')) {
        if (first && (word == "compact" || word == "bell"))
            f.ell = word == "bell";
        else if (!first && word == "column")
            f.column = true;
        else if (!first && word == "interleaved")
            f.interleaved = true;
        else
            fail("mock: matrix %s: unknown format '%s'", name.c_str(), text.c_str());
        first = false;
    }
    if (f.interleaved && !f.ell)
        fail("mock: matrix %s: only block ELL interleaves", name.c_str());
    return f;
}

class MockMatrix final : public Matrix {
public:
    MockMatrix(op_mat A, const std::string& format) : m_A{A}, m_format{parse_format(format, A->name)} {
        const int d = A->dim, rows = A->sparsity->rows->size;
        const Major major = m_format.column ? Major::column : Major::row;
        for (int p = 0; p < 2; p++) {
            const Part part = p == 0 ? Part::diag : Part::offd;
            mock::Part& mine = m_parts[p];
            mine.rows = rows;
            std::size_t blocks;
            if (m_format.ell) {
                BlockEll ell = bell_layout(A, part, p, 0, major, m_format.interleaved, -1);
                mine.width = ell.width;
                m_index[p][1] = upload(ell.col);
                blocks = (std::size_t)ell.width * rows;
            } else {
                BlockCsr csr = bcsr_layout(A, part, p, 0, major);
                m_index[p][0] = upload(csr.ptr);
                m_index[p][1] = upload(csr.col);
                blocks = csr.col.size();
            }
            mine.ptr = m_index[p][0];
            mine.col = m_index[p][1];
            // The library's own storage, which OP2 borrows.
            mine.values = mock::allocate(blocks * d * d);
            borrow(A, p, mine.values, blocks * d * d * sizeof(double));
        }
        if (A->sparsity->garray.empty())
            m_parts[1].rows = 0;
    }

    ~MockMatrix() override {
        for (int p = 0; p < 2; p++) {
            mock::free(m_parts[p].values);
            for (int *index : m_index[p])
                if (index != nullptr)
                    release(index);
        }
    }

    // The loops wrote the library's storage.
    void assemble() override {}

    const mock::Format& format() const { return m_format; }
    const mock::Part& local() const { return m_parts[0]; }
    const mock::Part& nonlocal() const { return m_parts[1]; }

private:
    op_mat m_A;
    mock::Format m_format;
    mock::Part m_parts[2];
    int *m_index[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
};

// The values of a vector's halo: each off-diagonal block column's d values
// from the rank that owns it, in garray order.
class Exchange {
public:
    Exchange(op_sparsity sp, int d) : m_d{d} {
        MPI_Comm c = comm();
        int ranks;
        MPI_Comm_size(c, &ranks);
        std::vector<long long> starts(ranks + 1, 0);
        long long start = sp->row_start;
        MPI_Allgather(&start, 1, MPI_LONG_LONG, starts.data(), 1, MPI_LONG_LONG, c);
        starts[ranks] = sp->global_rows;

        // What this rank wants of each, in garray order (by owner, ascending).
        m_recv_counts.assign(ranks, 0);
        std::vector<int> wanted;
        for (idx_g_t g : sp->garray) {
            const int owner = (int)(std::upper_bound(starts.begin(), starts.end(), (long long)g) - starts.begin()) - 1;
            m_recv_counts[owner]++;
            wanted.push_back((int)(g - starts[owner]));
        }
        m_send_counts.assign(ranks, 0);
        MPI_Alltoall(m_recv_counts.data(), 1, MPI_INT, m_send_counts.data(), 1, MPI_INT, c);
        auto displacements = [](const std::vector<int>& counts) {
            std::vector<int> displs(counts.size() + 1, 0);
            for (std::size_t r = 0; r < counts.size(); r++)
                displs[r + 1] = displs[r] + counts[r];
            return displs;
        };
        m_recv_displs = displacements(m_recv_counts);
        m_send_displs = displacements(m_send_counts);
        m_send.resize(m_send_displs[ranks]);
        MPI_Alltoallv(wanted.data(), m_recv_counts.data(), m_recv_displs.data(), MPI_INT, m_send.data(),
                      m_send_counts.data(), m_send_displs.data(), MPI_INT, c);

        // The same exchange, d values a block.
        for (auto *counts : {&m_send_counts, &m_recv_counts, &m_send_displs, &m_recv_displs})
            for (int& v : *counts)
                v *= d;
        m_rows = sp->rows->size * d;
        m_host.resize(m_rows);
        m_out.resize((std::size_t)m_send.size() * d);
        m_in.resize(sp->garray.size() * d);
    }

    void halo(const double *x, double *halo) {
        if (m_in.empty() && m_out.empty())
            return;
        mock::to_host(m_host.data(), x, sizeof(double) * m_rows);
        for (std::size_t k = 0; k < m_send.size(); k++)
            for (int c = 0; c < m_d; c++)
                m_out[k * m_d + c] = m_host[(std::size_t)m_send[k] * m_d + c];
        MPI_Alltoallv(m_out.data(), m_send_counts.data(), m_send_displs.data(), MPI_DOUBLE, m_in.data(),
                      m_recv_counts.data(), m_recv_displs.data(), MPI_DOUBLE, comm());
        mock::from_host(halo, m_in.data(), sizeof(double) * m_in.size());
    }

    std::size_t size() const { return m_in.size(); }

private:
    int m_d, m_rows = 0;
    std::vector<int> m_send, m_send_counts, m_recv_counts, m_send_displs, m_recv_displs;
    std::vector<double> m_host, m_out, m_in;
};

double global_dot(int n, const double *a, const double *b) {
    double local = mock::dot(n, a, b), sum;
    MPI_Allreduce(&local, &sum, 1, MPI_DOUBLE, MPI_SUM, comm());
    return sum;
}

class MockSolver final : public Solver {
public:
    explicit MockSolver(op_solver s) : m_s{s}, m_exchange{s->A->sparsity, s->A->dim} {
        const Options& o = s->options;
        if (o.method != "cg")
            fail("mock: solver %s: method '%s', it has cg", s->name.c_str(), o.method.c_str());
        if (o.pc != "jacobi" && o.pc != "none" && o.pc != "custom")
            fail("mock: solver %s: pc '%s', it has jacobi, none or a par-loop one", s->name.c_str(), o.pc.c_str());
        m_n = s->A->sparsity->rows->size * s->A->dim;
        for (double *&v : m_work)
            v = mock::allocate(m_n);
        m_inv = mock::allocate(m_n);
        m_halo = mock::allocate(m_exchange.size());
    }

    ~MockSolver() override {
        for (double *v : m_work)
            mock::free(v);
        mock::free(m_inv);
        mock::free(m_halo);
    }

    bool solve(op_dat b, op_dat x_dat) override {
        const Options& o = m_s->options;
        const double *bv = (const double *)memory(b);
        double *x = (double *)memory(x_dat);
        double *r = m_work[0], *z = m_work[1], *p = m_work[2], *q = m_work[3];
        op_mat P = m_s->P != nullptr ? m_s->P : m_s->A;
        if (o.pc == "jacobi" && m_pc_state != P->state) {
            const MockMatrix& pm = matrix(P);
            mock::inverse_diagonal(pm.format(), P->dim, m_n, pm.local(), m_inv);
            m_pc_state = P->state;
        }

        apply(x, r);
        mock::xpay(m_n, bv, -1.0, r);
        const double bnorm = std::sqrt(global_dot(m_n, bv, bv));
        precondition(r, z);
        copy(p, z, sizeof(double) * m_n);
        double rz = global_dot(m_n, r, z), rnorm = std::sqrt(global_dot(m_n, r, r));
        int its = 0;
        const double target = std::max(o.rtol * bnorm, o.atol);
        while (rnorm > target && its < o.maxit) {
            apply(p, q);
            const double alpha = rz / global_dot(m_n, p, q);
            mock::axpy(m_n, alpha, p, x);
            mock::axpy(m_n, -alpha, q, r);
            rnorm = std::sqrt(global_dot(m_n, r, r));
            its++;
            if (o.monitor)
                op_printf("mock %s: %d %.6e\n", m_s->name.c_str(), its, bnorm > 0 ? rnorm / bnorm : rnorm);
            if (rnorm <= target)
                break;
            precondition(r, z);
            const double rz_next = global_dot(m_n, r, z);
            mock::xpay(m_n, z, rz_next / rz, p);
            rz = rz_next;
        }
        iterations = its;
        residual = bnorm > 0 ? rnorm / bnorm : rnorm;
        return rnorm <= target;
    }

private:
    static const MockMatrix& matrix(op_mat A) { return *static_cast<const MockMatrix *>(A->lib.get()); }

    // y = A x, the assembled matrix or the par-loop operator.
    void apply(const double *x, double *y) {
        if (m_s->op != nullptr) {
            apply_through(m_s, m_s->op, m_s->op_ctx, x, y);
            return;
        }
        const MockMatrix& A = matrix(m_s->A);
        m_exchange.halo(x, m_halo);
        mock::spmv(A.format(), m_s->A->dim, m_n, A.local(), A.nonlocal(), x, m_halo, y);
    }

    void precondition(const double *r, double *z) {
        const Options& o = m_s->options;
        if (o.pc == "custom")
            apply_through(m_s, m_s->pc, m_s->pc_ctx, r, z);
        else if (o.pc == "jacobi")
            mock::multiply(m_n, m_inv, r, z);
        else
            copy(z, r, sizeof(double) * m_n);
    }

    op_solver m_s;
    Exchange m_exchange;
    int m_n = 0;
    double *m_work[4] = {nullptr, nullptr, nullptr, nullptr};
    double *m_inv = nullptr, *m_halo = nullptr;
    int m_pc_state = -1;
};

class MockLibrary final : public Library {
public:
    Order order() const override { return Order::sorted; }

    std::unique_ptr<Matrix> matrix(op_mat A, const std::string& format) override {
        return std::make_unique<MockMatrix>(A, format);
    }

    std::unique_ptr<Solver> solver(op_solver s) override { return std::make_unique<MockSolver>(s); }
};

} // namespace

std::unique_ptr<Library> make_mock(int, char **) { return std::make_unique<MockLibrary>(); }

} // namespace op::solve
