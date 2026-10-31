// The rocALUTION backend: a GlobalMatrix whose interior and ghost parts
// rocALUTION holds in the format asked for ("csr", "mcsr", "bcsr", "coo",
// "ell"), converted by rocALUTION itself from the pattern; the loops write
// its arrays in place, each part's layout checked against rocALUTION's own
// index arrays.  CG, BiCGStab, GMRES or FGMRES over GlobalVectors, with
// Jacobi, pairwise AMG, none or a par-loop preconditioner.  Host variants,
// and HIP ones in a HIP build of rocALUTION.

#include <op2/solve/solve.h>

#include <rocalution/rocalution.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace op::solve {

namespace {

namespace roc = rocalution;
using matrix_type = roc::GlobalMatrix<double>;
using vector_type = roc::GlobalVector<double>;

// The parallel manager of a sparsity's rows: its scalar rows and columns,
// the rows each rank sends (boundary), and the ghost columns, in garray
// order, from each rank that owns them.
struct Manager {
    roc::ParallelManager pm;
    MPI_Comm c;

    Manager(op_sparsity sp, int d) : c{comm()} {
        int ranks;
        MPI_Comm_size(c, &ranks);
        std::vector<long long> starts(ranks + 1, 0);
        long long start = sp->row_start;
        MPI_Allgather(&start, 1, MPI_LONG_LONG, starts.data(), 1, MPI_LONG_LONG, c);
        starts[ranks] = sp->global_rows;

        std::vector<int> recv_counts(ranks, 0), wanted;
        for (idx_g_t g : sp->garray) {
            const int owner = (int)(std::upper_bound(starts.begin(), starts.end(), (long long)g) - starts.begin()) - 1;
            recv_counts[owner]++;
            wanted.push_back((int)(g - starts[owner]));
        }
        std::vector<int> send_counts(ranks, 0);
        MPI_Alltoall(recv_counts.data(), 1, MPI_INT, send_counts.data(), 1, MPI_INT, c);
        std::vector<int> recv_displs(ranks + 1, 0), send_displs(ranks + 1, 0);
        for (int r = 0; r < ranks; r++) {
            recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
            send_displs[r + 1] = send_displs[r] + send_counts[r];
        }
        std::vector<int> send(send_displs[ranks]);
        MPI_Alltoallv(wanted.data(), recv_counts.data(), recv_displs.data(), MPI_INT, send.data(),
                      send_counts.data(), send_displs.data(), MPI_INT, c);

        // In scalar rows: d per block.
        for (int b : send)
            for (int k = 0; k < d; k++)
                m_boundary.push_back(b * d + k);
        for (int r = 0; r < ranks; r++) {
            if (recv_counts[r] > 0) {
                m_recvs.push_back(r);
                m_recv_offsets.push_back(recv_displs[r] * d);
            }
            if (send_counts[r] > 0) {
                m_sends.push_back(r);
                m_send_offsets.push_back(send_displs[r] * d);
            }
        }
        m_recv_offsets.push_back(recv_displs[ranks] * d);
        m_send_offsets.push_back(send_displs[ranks] * d);

        pm.SetMPICommunicator(&c);
        pm.SetGlobalNrow(sp->global_rows * d);
        pm.SetGlobalNcol(sp->global_cols * d);
        pm.SetLocalNrow(sp->rows->size * d);
        pm.SetLocalNcol(sp->cols->size * d);
        pm.SetBoundaryIndex((int)m_boundary.size(), m_boundary.data());
        pm.SetReceivers((int)m_recvs.size(), m_recvs.data(), m_recv_offsets.data());
        pm.SetSenders((int)m_sends.size(), m_sends.data(), m_send_offsets.data());
    }

private:
    std::vector<int> m_boundary, m_recvs, m_recv_offsets, m_sends, m_send_offsets;
};

template<typename T>
T *roc_copy(const std::vector<T>& values) {
    T *out = nullptr;
    roc::allocate_host((int64_t)values.size(), &out);
    std::copy(values.begin(), values.end(), out);
    return out;
}

bool same(const int *a, const std::vector<int>& b) { return std::equal(b.begin(), b.end(), a); }

class RocMatrix final : public Matrix {
public:
    RocMatrix(op_mat A, const std::string& format) : m_A{A}, m_manager{A->sparsity, A->dim} {
        if (on_device)
            fail("rocALUTION: matrix %s: a device variant needs a HIP build of rocALUTION", A->name.c_str());
        op_sparsity sp = A->sparsity;
        const int d = A->dim;
        unsigned int target = roc::CSR;
        if (format == "mcsr")
            target = roc::MCSR;
        else if (format == "bcsr")
            target = roc::BCSR;
        else if (format == "coo")
            target = roc::COO;
        else if (format == "ell")
            target = roc::ELL;
        else if (!format.empty() && format != "csr")
            fail("rocALUTION: matrix %s: unknown format '%s'", A->name.c_str(), format.c_str());

        // The pattern as CSR in rocALUTION's own memory, which it converts.
        const ScalarCsr dsc = csr_layout(A, Part::diag, 0, 0);
        const ScalarCsr osc = csr_layout(A, Part::offd, 1, 0);
        int *lptr = roc_copy(dsc.ptr), *lcol = roc_copy(dsc.col), *gptr = roc_copy(osc.ptr), *gcol = roc_copy(osc.col);
        double *lval = roc_copy(std::vector<double>(dsc.col.size(), 0.0));
        double *gval = roc_copy(std::vector<double>(osc.col.size(), 0.0));
        m_matrix.SetParallelManager(m_manager.pm);
        m_matrix.SetDataPtrCSR(&lptr, &lcol, &lval, &gptr, &gcol, &gval, A->name, (int64_t)dsc.col.size(),
                               (int64_t)osc.col.size());
        // rocALUTION takes no 1 x 1 blocks: a scalar matrix stays CSR.
        if (target == roc::BCSR && d == 1)
            target = roc::CSR;
        if (target != roc::CSR)
            m_matrix.ConvertTo(target, target == roc::BCSR ? d : 1);
        locate(Part::diag, const_cast<roc::LocalMatrix<double>&>(m_matrix.GetInterior()));
        if (!sp->garray.empty())
            locate(Part::offd, const_cast<roc::LocalMatrix<double>&>(m_matrix.GetGhost()));
    }

    // The loops wrote rocALUTION's arrays.
    void assemble() override {}

    matrix_type& matrix() { return m_matrix; }
    roc::ParallelManager& manager() { return m_manager.pm; }

private:
    // A part's arrays as rocALUTION holds them: its values borrowed, the
    // layout found from, or checked against, its index arrays.
    void locate(Part part, roc::LocalMatrix<double>& m) {
        const int region = part == Part::diag ? 0 : 1, d = m_A->dim;
        const int nrow = (int)m.GetM(), ncol = (int)m.GetN();
        const int64_t nnz = m.GetNnz();
        const std::string name = m_A->name + (part == Part::diag ? "_interior" : "_ghost");
        auto unexpected = [&](const char *what) {
            fail("rocALUTION: matrix %s: its %s part's %s is not laid out as expected", m_A->name.c_str(),
                 part == Part::diag ? "interior" : "ghost", what);
        };
        switch (m.GetFormat()) {
        case roc::CSR: {
            int *ptr = nullptr, *col = nullptr;
            double *val = nullptr;
            m.LeaveDataPtrCSR(&ptr, &col, &val);
            csr_located(m_A, part, region, std::vector<int>(ptr, ptr + nrow + 1), std::vector<int>(col, col + nnz));
            double *values = val; // SetDataPtr* takes it, leaving val null
            m.SetDataPtrCSR(&ptr, &col, &val, name, nnz, nrow, ncol);
            borrow(m_A, region, m_values[region] = values, nnz * sizeof(double));
            break;
        }
        case roc::COO: {
            int *row = nullptr, *col = nullptr;
            double *val = nullptr;
            m.LeaveDataPtrCOO(&row, &col, &val);
            std::vector<int> ptr(nrow + 1, 0);
            for (int64_t k = 0; k < nnz; k++)
                ptr[row[k] + 1]++;
            for (int i = 0; i < nrow; i++)
                ptr[i + 1] += ptr[i];
            csr_located(m_A, part, region, ptr, std::vector<int>(col, col + nnz));
            double *values = val;
            m.SetDataPtrCOO(&row, &col, &val, name, nnz, nrow, ncol);
            borrow(m_A, region, m_values[region] = values, nnz * sizeof(double));
            break;
        }
        case roc::MCSR: {
            int *ptr = nullptr, *col = nullptr;
            double *val = nullptr;
            m.LeaveDataPtrMCSR(&ptr, &col, &val);
            // Row i's off-diagonal entries from nrow + its CSR start - i, its
            // diagonal at i.
            const ScalarCsr expected = csr_layout(m_A, part, region, nrow, Diagonal::separate, 0);
            for (int i = 0; i <= nrow; i++)
                if (ptr[i] != nrow + expected.ptr[i])
                    unexpected("rows");
            if (!std::equal(expected.col.begin(), expected.col.end(), col + nrow))
                unexpected("columns");
            double *values = val;
            m.SetDataPtrMCSR(&ptr, &col, &val, name, nnz, nrow, ncol);
            borrow(m_A, region, m_values[region] = values, nnz * sizeof(double));
            break;
        }
        case roc::BCSR: {
            int *ptr = nullptr, *col = nullptr, blockdim;
            double *val = nullptr;
            m.LeaveDataPtrBCSR(&ptr, &col, &val, blockdim);
            const BlockCsr expected = bcsr_layout(m_A, part, region, 0, Major::column);
            if (blockdim != d || !same(ptr, expected.ptr) || !same(col, expected.col))
                unexpected("blocks");
            const int64_t blocks = (int64_t)expected.col.size();
            double *values = val;
            m.SetDataPtrBCSR(&ptr, &col, &val, name, blocks, nrow / d, ncol / d, blockdim);
            borrow(m_A, region, m_values[region] = values, blocks * d * d * sizeof(double));
            break;
        }
        case roc::ELL: {
            int *col = nullptr, width;
            double *val = nullptr;
            m.LeaveDataPtrELL(&col, &val, width);
            const Ell expected = ell_layout(m_A, part, region, 0, -1);
            if (width != expected.width || !same(col, expected.col))
                unexpected("columns");
            double *values = val;
            m.SetDataPtrELL(&col, &val, name, (int64_t)width * nrow, nrow, ncol, width);
            borrow(m_A, region, m_values[region] = values, (std::size_t)width * nrow * sizeof(double));
            break;
        }
        default:
            fail("rocALUTION: matrix %s: rocALUTION kept its %s part in a format OP2 does not lay out",
                 m_A->name.c_str(), part == Part::diag ? "interior" : "ghost");
        }
    }

    op_mat m_A;
    Manager m_manager;
    matrix_type m_matrix;
    double *m_values[2] = {nullptr, nullptr};
};

RocMatrix& roc_matrix(op_mat A) { return *static_cast<RocMatrix *>(A->lib.get()); }

// A par-loop preconditioner as rocALUTION takes one.
class Callback : public roc::Preconditioner<matrix_type, vector_type, double> {
public:
    explicit Callback(op_solver s) : m_s{s} {}

    void Print() const override {}
    void Build() override { this->build_ = true; }

    void Solve(const vector_type& rhs, vector_type *x) override {
        const std::size_t n = (std::size_t)m_s->A->sparsity->rows->size * m_s->A->dim;
        m_r.resize(n);
        m_z.resize(n);
        rhs.GetInterior().CopyToData(m_r.data());
        apply_through(m_s, m_s->pc, m_s->pc_ctx, m_r.data(), m_z.data());
        x->GetInterior().CopyFromData(m_z.data());
    }

protected:
    void PrintStart_() const override {}
    void PrintEnd_() const override {}
    void MoveToHostLocalData_() override {}
    void MoveToAcceleratorLocalData_() override {}

private:
    op_solver m_s;
    std::vector<double> m_r, m_z;
};

class RocSolver final : public Solver {
public:
    explicit RocSolver(op_solver s) : m_s{s} {
        const Options& o = s->options;
        if (s->op != nullptr)
            fail("rocALUTION: solver %s: a par-loop operator is not supported", s->name.c_str());
        if (o.method == "cg")
            m_ls = std::make_unique<roc::CG<matrix_type, vector_type, double>>();
        else if (o.method == "bicgstab")
            m_ls = std::make_unique<roc::BiCGStab<matrix_type, vector_type, double>>();
        else if (o.method == "gmres") {
            auto gmres = std::make_unique<roc::GMRES<matrix_type, vector_type, double>>();
            gmres->SetBasisSize(o.restart);
            m_ls = std::move(gmres);
        } else if (o.method == "fgmres") {
            auto fgmres = std::make_unique<roc::FGMRES<matrix_type, vector_type, double>>();
            fgmres->SetBasisSize(o.restart);
            m_ls = std::move(fgmres);
        } else
            fail("rocALUTION: unknown method '%s'", o.method.c_str());

        if (o.pc == "jacobi")
            m_pc = std::make_unique<roc::Jacobi<matrix_type, vector_type, double>>();
        else if (o.pc == "amg")
            m_pc = std::make_unique<roc::PairwiseAMG<matrix_type, vector_type, double>>();
        else if (o.pc == "custom")
            m_pc = std::make_unique<Callback>(s);
        else if (o.pc != "none")
            fail("rocALUTION: unknown pc '%s'", o.pc.c_str());

        RocMatrix& A = roc_matrix(s->P != nullptr ? s->P : s->A);
        m_ls->SetOperator(roc_matrix(s->A).matrix());
        if (m_pc)
            m_ls->SetPreconditioner(*m_pc);
        m_ls->Init(o.atol, o.rtol, 1e100, o.maxit);
        m_ls->Verbose(o.monitor ? 2 : 0);
        for (vector_type *v : {&m_b, &m_x}) {
            v->SetParallelManager(A.manager());
            v->Allocate(s->name, (int64_t)s->A->sparsity->global_rows * s->A->dim);
        }
    }

    bool solve(op_dat b, op_dat x) override {
        op_mat P = m_s->P != nullptr ? m_s->P : m_s->A;
        if (!m_built) {
            m_ls->Build();
            m_built = true;
        } else if (m_state != P->state)
            m_ls->ReBuildNumeric();
        m_state = P->state;
        m_b.GetInterior().CopyFromData((const double *)memory(b));
        m_x.GetInterior().CopyFromData((const double *)memory(x));
        m_ls->Solve(m_b, &m_x);
        m_x.GetInterior().CopyToData((double *)memory(x));
        iterations = m_ls->GetIterationCount();
        const double bnorm = m_b.Norm(), rnorm = m_ls->GetCurrentResidual();
        residual = bnorm > 0 ? rnorm / bnorm : rnorm;
        return m_ls->GetSolverStatus() != 3 && iterations < m_s->options.maxit;
    }

private:
    op_solver m_s;
    // The preconditioner outlives the solver, which refers to it.
    std::unique_ptr<roc::Solver<matrix_type, vector_type, double>> m_pc;
    std::unique_ptr<roc::IterativeLinearSolver<matrix_type, vector_type, double>> m_ls;
    vector_type m_b, m_x;
    bool m_built = false;
    int m_state = -1;
};

class RocLibrary final : public Library {
public:
    RocLibrary() {
        int rank;
        MPI_Comm_rank(comm(), &rank);
        roc::init_rocalution(rank);
    }

    ~RocLibrary() override { roc::stop_rocalution(); }

    Order order() const override { return Order::sorted; }

    std::unique_ptr<Matrix> matrix(op_mat A, const std::string& format) override {
        return std::make_unique<RocMatrix>(A, format);
    }

    std::unique_ptr<Solver> solver(op_solver s) override { return std::make_unique<RocSolver>(s); }
};

} // namespace

std::unique_ptr<Library> make_rocalution(int, char **) { return std::make_unique<RocLibrary>(); }

} // namespace op::solve
