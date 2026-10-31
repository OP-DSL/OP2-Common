#include <op2/solve/solve.h>

#include <op2/env.h>
#include <op_lib_c.h>

#ifdef OPMPI
#include <op_mpi_core.h>
#endif

#if OP2_SOLVE_DEVICE
#include <op2/gpu/check.h>
#include <op2/gpu/shims.h>
#include <op_cuda_rt_support.h>
#endif

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace op::solve {

namespace {

std::unique_ptr<Library> library;
bool finalise_mpi = false;
std::vector<op_sparsity> sparsities;
std::vector<op_mat> mats;
std::vector<op_solver> solvers;

} // namespace

void fail(char const *format, ...) {
    va_list args;
    va_start(args, format);
    std::fprintf(stderr, "op_solve: ");
    std::vfprintf(stderr, format, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    int initialised;
    MPI_Initialized(&initialised);
    if (initialised)
        MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

MPI_Comm comm() {
#ifdef OPMPI
    return OP_MPI_WORLD;
#else
    return MPI_COMM_SELF;
#endif
}

Settings settings;

void read_env() {
    env::parse_text("OP_SOLVE_LIB", settings.lib);
    env::parse_text("OP_SOLVE_FORMAT", settings.format);
    env::parse_bool("OP_SOLVE_MEMORY", settings.memory);
    env::parse_number("OP_SOLVE_HYPRE_POOL", settings.hypre_pool);
}

Options parse_options(char const *text) {
    Options options;
    std::string all = text != nullptr ? text : "";
    std::size_t split = all.find("--");
    if (split != std::string::npos) {
        options.library = all.substr(split + 2);
        all.resize(split);
    }

    std::istringstream words(all);
    std::string word;
    while (words >> word) {
        std::size_t eq = word.find('=');
        if (eq == std::string::npos)
            fail("option '%s' is not key=value", word.c_str());
        std::string key = word.substr(0, eq), value = word.substr(eq + 1);
        if (key == "method")
            options.method = value;
        else if (key == "pc")
            options.pc = value;
        else if (key == "rtol")
            options.rtol = std::atof(value.c_str());
        else if (key == "atol")
            options.atol = std::atof(value.c_str());
        else if (key == "maxit")
            options.maxit = std::atoi(value.c_str());
        else if (key == "restart")
            options.restart = std::atoi(value.c_str());
        else if (key == "monitor")
            options.monitor = std::atoi(value.c_str()) != 0;
        else
            fail("unknown option '%s'", key.c_str());
    }
    return options;
}

char *memory(op_dat dat) { return on_device ? dat->data_d : dat->data; }

void copy(void *dst, const void *src, std::size_t bytes) {
#if OP2_SOLVE_DEVICE
    GPU_CHECK(gpuMemcpy(dst, src, bytes, gpuMemcpyDeviceToDevice));
#else
    std::memcpy(dst, src, bytes);
#endif
}

void download(void *dst, const void *src, std::size_t bytes) {
#if OP2_SOLVE_DEVICE
    GPU_CHECK(gpuMemcpy(dst, src, bytes, gpuMemcpyDeviceToHost));
#else
    std::memcpy(dst, src, bytes);
#endif
}

void zero(void *dst, std::size_t bytes) {
#if OP2_SOLVE_DEVICE
    GPU_CHECK(gpuMemset(dst, 0, bytes));
#else
    std::memset(dst, 0, bytes);
#endif
}

void report_memory(char const *stage, std::size_t own) {
    if (!settings.memory)
        return;
    const double mib = 1024.0 * 1024.0;
#if OP2_SOLVE_DEVICE
    std::size_t free, total;
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemGetInfo(&free, &total));
    op_printf("op_solve memory: %-28s %10.1f MiB own, %10.1f MiB device in use\n", stage, own / mib,
              (total - free) / mib);
#else
    op_printf("op_solve memory: %-28s %10.1f MiB own\n", stage, own / mib);
#endif
}

#if !OP2_SOLVE_DEVICE
int *upload(const std::vector<int>& values) {
    int *host = (int *)std::malloc(sizeof(int) * values.size() + 1);
    std::memcpy(host, values.data(), sizeof(int) * values.size());
    return host;
}

void release(void *memory) { std::free(memory); }
#endif

void enter(op_set set, op_arg *args, int nargs) {
    op_mpi_halo_exchanges(set, nargs, args, on_device ? 2 : 1);
    op_mpi_wait_all(nargs, args);
}

void leave(op_arg *args, int nargs) {
    if (on_device)
        op_mpi_set_dirtybit_cuda(nargs, args);
    else
        op_mpi_set_dirtybit(nargs, args);
}

void apply_through(op_solver s, op_apply_fn apply, void *ctx, const double *r, double *z) {
    const std::size_t bytes = (std::size_t)s->in->set->size * s->in->size;
    op_arg in_arg = op_arg_dat_core(s->in, -1, OP_ID, s->in->dim, "double", OP_WRITE);
    enter(s->in->set, &in_arg, 1);
    copy(memory(s->in), r, bytes);
    leave(&in_arg, 1);

    apply(s->in, s->out, ctx);

    op_arg out_arg = op_arg_dat_core(s->out, -1, OP_ID, s->out->dim, "double", OP_READ);
    enter(s->out->set, &out_arg, 1);
    copy(z, memory(s->out), bytes);
}

namespace {

// The global number of each element of set the rank holds, halo included:
// its owner's offset plus its index there.
std::vector<idx_g_t> global_numbers(op_set set, op_map reach, idx_g_t start) {
    const int held = set->size + set->exec_size + set->nonexec_size;
    std::vector<idx_g_t> numbers(held);
    for (int i = 0; i < set->size; i++)
        numbers[i] = start + i;
    if (held == set->size)
        return numbers;
    if (reach == nullptr)
        fail("set %s has a halo but no map reaches it", set->name);

    std::string name = std::string("op_solve_gid_") + set->name;
    op_dat gid = op_decl_dat_temp_char(set, 1, "long long", sizeof(idx_g_t), name.c_str());
    std::memcpy(gid->data, numbers.data(), sizeof(idx_g_t) * set->size);
    gid->dirtybit = 1;
    gid->dirty_hd = 1;

    op_arg arg = op_arg_dat_core(gid, 0, reach, 1, "long long", OP_READ);
    op_mpi_halo_exchanges(reach->from, 1, &arg, 1);
    op_mpi_wait_all(1, &arg);
    std::memcpy(numbers.data(), gid->data, sizeof(idx_g_t) * held);
    op_free_dat_temp_char(gid);
    return numbers;
}

idx_g_t exclusive_sum(idx_g_t value, idx_g_t *total) {
    idx_g_t start = 0;
    MPI_Exscan(&value, &start, 1, MPI_LONG_LONG, MPI_SUM, comm());
    int rank;
    MPI_Comm_rank(comm(), &rank);
    if (rank == 0)
        start = 0;
    MPI_Allreduce(&value, total, 1, MPI_LONG_LONG, MPI_SUM, comm());
    return start;
}

// The executed elements of a pair's set: an increment loop runs over the
// exec halo too.
int executed(op_set from) { return from->size + from->exec_size; }

int row_of(const op_sparsity_core::Pair& pair, int e, int i) {
    return pair.rowmap != nullptr ? pair.rowmap->map[(std::size_t)e * pair.rowdim + i] : e;
}

int col_of(const op_sparsity_core::Pair& pair, int e, int j) {
    return pair.colmap != nullptr ? pair.colmap->map[(std::size_t)e * pair.coldim + j] : e;
}

void upload_map(op_map map) {
#if OP2_SOLVE_DEVICE
    const int elements = executed(map->from), stride = (elements + 31) & ~31;
    std::vector<int> transposed((std::size_t)map->dim * stride, 0);
    for (int d = 0; d < map->dim; d++)
        for (int e = 0; e < elements; e++)
            transposed[(std::size_t)d * stride + e] = map->map[(std::size_t)e * map->dim + d];
    int *host = transposed.data();
    op_cpHostToDevice((void **)&map->map_d, (void **)&host, sizeof(int) * transposed.size());
#else
    (void)map;
#endif
}

void build(op_sparsity sp, Order order) {
    op_set rows = sp->rows, cols = sp->cols;
    const int nrows = rows->size, nowned = cols->size;
    const bool square = rows == cols;

    sp->row_start = exclusive_sum(rows->size, &sp->global_rows);
    sp->col_start = exclusive_sum(cols->size, &sp->global_cols);

    op_map reach = nullptr;
    for (auto& pair : sp->pairs)
        if (pair.colmap != nullptr)
            reach = pair.colmap;
    const std::vector<idx_g_t> gid = global_numbers(cols, reach, sp->col_start);

    // Every row's columns, as local indices (halo ones past the owned).
    std::vector<int> ptr(nrows + 1, 0);
    for (auto& pair : sp->pairs)
        for (int e = 0; e < executed(pair.from); e++)
            for (int i = 0; i < pair.rowdim; i++) {
                int I = row_of(pair, e, i);
                if (I < nrows)
                    ptr[I + 1] += pair.coldim;
            }
    for (int I = 0; I < nrows; I++)
        ptr[I + 1] += ptr[I];

    std::vector<int> col(ptr[nrows]);
    std::vector<int> fill(ptr.begin(), ptr.end() - 1);
    for (auto& pair : sp->pairs)
        for (int e = 0; e < executed(pair.from); e++)
            for (int i = 0; i < pair.rowdim; i++) {
                int I = row_of(pair, e, i);
                if (I < nrows)
                    for (int j = 0; j < pair.coldim; j++)
                        col[fill[I]++] = col_of(pair, e, j);
            }

    // Halo columns by global number, then each row split into its parts.
    std::vector<idx_g_t>& garray = sp->garray;
    for (int k = 0; k < ptr[nrows]; k++)
        if (col[k] >= nowned)
            garray.push_back(gid[col[k]]);
    std::sort(garray.begin(), garray.end());
    garray.erase(std::unique(garray.begin(), garray.end()), garray.end());

    sp->diag_ptr.assign(nrows + 1, 0);
    sp->offd_ptr.assign(nrows + 1, 0);
    std::vector<int> row_diag, row_offd;
    for (int I = 0; I < nrows; I++) {
        row_diag.clear();
        row_offd.clear();
        for (int k = ptr[I]; k < ptr[I + 1]; k++) {
            if (col[k] < nowned)
                row_diag.push_back(col[k]);
            else
                row_offd.push_back(
                    (int)(std::lower_bound(garray.begin(), garray.end(), gid[col[k]]) - garray.begin()));
        }
        std::sort(row_diag.begin(), row_diag.end());
        row_diag.erase(std::unique(row_diag.begin(), row_diag.end()), row_diag.end());
        std::sort(row_offd.begin(), row_offd.end());
        row_offd.erase(std::unique(row_offd.begin(), row_offd.end()), row_offd.end());
        if (order == Order::diagonal_first && square) {
            auto diag = std::find(row_diag.begin(), row_diag.end(), I);
            std::rotate(row_diag.begin(), diag, diag + 1);
        }
        sp->diag_col.insert(sp->diag_col.end(), row_diag.begin(), row_diag.end());
        sp->offd_col.insert(sp->offd_col.end(), row_offd.begin(), row_offd.end());
        sp->diag_ptr[I + 1] = (int)sp->diag_col.size();
        sp->offd_ptr[I + 1] = (int)sp->offd_col.size();
    }
    sp->diag_blocks = (int)sp->diag_col.size();
    sp->offd_blocks = (int)sp->offd_col.size();

    // Halo column index into garray, and each pair's slot maps.
    const int nheld = cols->size + cols->exec_size + cols->nonexec_size;
    std::vector<int> halo_col(nheld, -1);
    for (int J = nowned; J < nheld; J++) {
        auto at = std::lower_bound(garray.begin(), garray.end(), gid[J]);
        if (at != garray.end() && *at == gid[J])
            halo_col[J] = (int)(at - garray.begin());
    }

    auto find = [](const std::vector<int>& v, int begin, int end, int value, bool sorted) {
        if (!sorted) {
            for (int k = begin; k < end; k++)
                if (v[k] == value)
                    return k;
            return -1;
        }
        auto at = std::lower_bound(v.begin() + begin, v.begin() + end, value);
        return at != v.begin() + end && *at == value ? (int)(at - v.begin()) : -1;
    };
    const bool sorted = order == Order::sorted;
    const int owned = sp->diag_blocks + sp->offd_blocks;

    int scratch = 0;
    std::vector<std::vector<int>> slot_values;
    for (auto& pair : sp->pairs)
        for (int i = 0; i < pair.rowdim; i++) {
            std::vector<int> slots((std::size_t)executed(pair.from) * pair.coldim);
            for (int e = 0; e < executed(pair.from); e++) {
                int I = row_of(pair, e, i);
                for (int j = 0; j < pair.coldim; j++) {
                    int J = col_of(pair, e, j), slot;
                    // Generated code takes a block as diagonal only through the
                    // same map and index on both sides.
                    if (I < nrows && J == I && !(pair.rowmap == pair.colmap && i == j))
                        sp->diagonal_elsewhere = true;
                    if (I >= nrows)
                        slot = owned + scratch++;
                    else if (J < nowned)
                        slot = find(sp->diag_col, sp->diag_ptr[I], sp->diag_ptr[I + 1], J, sorted);
                    else
                        slot = find(sp->offd_col, sp->offd_ptr[I], sp->offd_ptr[I + 1], halo_col[J], true);
                    if (slot < 0)
                        fail("sparsity %s: block (%d, %d) not in the pattern", sp->name.c_str(), I, J);
                    if (I < nrows && J >= nowned)
                        slot += sp->diag_blocks;
                    slots[(std::size_t)e * pair.coldim + j] = slot;
                }
            }
            slot_values.push_back(std::move(slots));
        }
    sp->scratch_blocks = scratch;

    sp->nnz = op_decl_set_core(owned + scratch, (sp->name + "_slots").c_str());
    std::size_t next = 0;
    for (auto& pair : sp->pairs)
        for (int i = 0; i < pair.rowdim; i++) {
            std::vector<int>& slots = slot_values[next++];
            std::string name = sp->name + "_slots_" + std::to_string(next);
            op_map map = op_decl_map_core(pair.from, sp->nnz, pair.coldim, nullptr, name.c_str());
            map->map = (int *)std::malloc(sizeof(int) * slots.size());
            std::memcpy(map->map, slots.data(), sizeof(int) * slots.size());
            map->user_managed = 0;
            pair.slots.push_back(map);
        }
    op_mpi_local_objects();
    std::size_t bytes = 0;
    for (auto& pair : sp->pairs)
        for (op_map map : pair.slots) {
            upload_map(map);
            bytes += sizeof(int) * (std::size_t)executed(map->from) * map->dim;
        }
    report_memory(("sparsity " + sp->name + " slot maps").c_str(), bytes);
}

// Whether slot s of A (in the diagonal part, row I) is a diagonal block.
bool diagonal_slot(op_sparsity sp, int I, int s) {
    return sp->rows == sp->cols && s < sp->diag_blocks && sp->diag_col[s] == I;
}

// Each slot's block's values, entry k (row major) of slot s at s * d * d + k
// of blocked, added into A's storage, or (gather) read from it: what
// untranslated loops reach, which run on the host.
template<bool Gather>
void untranslated_pass(op_mat A, double *blocked) {
    op_sparsity sp = A->sparsity;
    const int d = A->dim, bs = d * d, nrows = sp->rows->size;
    auto pass = [&](int s, bool diagonal) {
        const Entry& e = A->layout[s];
        double *region = (double *)A->regions[e[mat::aux] & mat::region_mask].data;
        for (int r = 0; r < d; r++)
            for (int c = 0; c < d; c++) {
                double& value = region[position(e, d, r, c, diagonal)];
                double& block = blocked[(std::size_t)s * bs + r * d + c];
                if (Gather)
                    block = value;
                else
                    value += block;
            }
    };
    for (int I = 0; I < nrows; I++) {
        for (int s = sp->diag_ptr[I]; s < sp->diag_ptr[I + 1]; s++)
            pass(s, diagonal_slot(sp, I, s));
        for (int k = sp->offd_ptr[I]; k < sp->offd_ptr[I + 1]; k++)
            pass(sp->diag_blocks + k, false);
    }
}

// op_mat_storage::untranslated: the blocks one after another, for a loop the
// translator never saw (op_seq.h).
char *untranslated(op_mat_storage *view, int acc) {
    op_mat A = nullptr;
    for (op_mat M : mats)
        if (&M->view == view)
            A = M;
    if (on_device)
        fail("matrix %s: untranslated loops run on the host, its storage is on the device", A->name.c_str());
    const std::size_t values = (std::size_t)A->sparsity->nnz->size * A->dim * A->dim;
    if (A->untranslated.size() != values)
        A->untranslated.assign(values, 0.0);
    if (acc == OP_READ) {
        if (A->pending)
            fail("matrix %s: read by a loop before op_mat_assemble took its increments", A->name.c_str());
        untranslated_pass<true>(A, A->untranslated.data());
    } else
        A->pending = true;
    return (char *)A->untranslated.data();
}

// After the library set up A's owned slots: scratch slots for blocks of halo
// rows in region 2, one after another, and the layout where loops read it.
void finish_storage(op_mat A) {
    op_sparsity sp = A->sparsity;
    const int bs = A->dim * A->dim, owned = sp->diag_blocks + sp->offd_blocks;
    if (sp->scratch_blocks > 0) {
        lend(A, 2, (std::size_t)sp->scratch_blocks * bs * sizeof(double));
        for (int k = 0; k < sp->scratch_blocks; k++)
            A->layout[owned + k] = {k * bs, A->dim, 1, 2};
    }
    for (int r = 0; r < mat::regions; r++)
        if (A->regions[r].bytes > INT_MAX * sizeof(double))
            fail("matrix %s: region %d holds %zu values, past the 32-bit offsets loops take", A->name.c_str(), r,
                 A->regions[r].bytes / sizeof(double));

    std::vector<int> flat(A->layout.size() * mat::entry);
    for (std::size_t s = 0; s < A->layout.size(); s++)
        std::copy(A->layout[s].begin(), A->layout[s].end(), flat.begin() + s * mat::entry);
    A->view.dim = A->dim;
    A->view.layout = flat.empty() ? nullptr : (const int *)A->layout.data()->data();
    if (on_device) {
        A->layout_d = upload(flat);
        A->view.layout_d = A->layout_d;
    }
    for (int r = 0; r < mat::regions; r++)
        (on_device ? A->view.region_d : A->view.region)[r] = A->regions[r].data;
    A->view.untranslated = untranslated;
}

} // namespace

char *lend(op_mat A, int region, std::size_t bytes) {
    Region& r = A->regions[region];
    if (r.data != nullptr)
        fail("matrix %s: region %d set twice", A->name.c_str(), region);
#if OP2_SOLVE_DEVICE
    GPU_CHECK(gpuMalloc((void **)&r.data, bytes + 1));
    GPU_CHECK(gpuMemset(r.data, 0, bytes + 1));
#else
    r.data = (char *)std::calloc(bytes + 1, 1);
#endif
    r.bytes = bytes;
    r.owned = true;
    return r.data;
}

void borrow(op_mat A, int region, void *data, std::size_t bytes) {
    Region& r = A->regions[region];
    if (r.data != nullptr)
        fail("matrix %s: region %d set twice", A->name.c_str(), region);
    r = {(char *)data, bytes, false};
}

} // namespace op::solve

using namespace op::solve;

extern "C" {

void op_solve_init(int argc, char **argv, char const *lib) {
    int initialised;
    MPI_Initialized(&initialised);
    if (!initialised) {
        MPI_Init(&argc, &argv);
        finalise_mpi = true;
    }

    std::string name = lib != nullptr ? lib : settings.lib;
#ifdef OP2_SOLVE_PETSC
    if (name.empty() || name == "petsc")
        library = make_petsc(argc, argv);
#endif
#ifdef OP2_SOLVE_HYPRE
    if (!library && (name.empty() || name == "hypre"))
        library = make_hypre(argc, argv);
#endif
#ifdef OP2_SOLVE_GINKGO
    if (!library && (name.empty() || name == "ginkgo"))
        library = make_ginkgo(argc, argv);
#endif
#ifdef OP2_SOLVE_AMGX
    if (!library && (name.empty() || name == "amgx"))
        library = make_amgx(argc, argv);
#endif
#ifdef OP2_SOLVE_ROCALUTION
    if (!library && (name.empty() || name == "rocalution"))
        library = make_rocalution(argc, argv);
#endif
#ifdef OP2_SOLVE_MOCK
    if (!library && name == "mock")
        library = make_mock(argc, argv);
#endif
    if (!library)
        fail("no solver library '%s' in this build", name.c_str());
}

void op_solve_exit(void) {
    for (op_solver s : solvers) {
        s->lib.reset();
        if (s->in != nullptr) {
            op_free_dat_temp_char(s->in);
            op_free_dat_temp_char(s->out);
        }
        delete s;
    }
    for (op_mat A : mats) {
        A->lib.reset();
        for (Region& r : A->regions)
            if (r.owned)
                release(r.data);
        if (A->layout_d != nullptr)
            release(A->layout_d);
        delete A;
    }
    for (op_sparsity sp : sparsities)
        delete sp;
    solvers.clear();
    mats.clear();
    sparsities.clear();
    library.reset();
    if (finalise_mpi)
        MPI_Finalize();
}

op_sparsity op_decl_sparsity(op_set rows, op_set cols, int npairs, op_map *rowmaps, op_map *colmaps,
                             char const *name) {
    if (!library)
        fail("op_decl_sparsity before op_solve_init");
    op_sparsity sp = new op_sparsity_core{};
    sp->rows = rows;
    sp->cols = cols;
    sp->name = name;
    for (int k = 0; k < npairs; k++) {
        op_map rowmap = rowmaps[k], colmap = colmaps[k];
        op_set from = rowmap != nullptr ? rowmap->from : colmap != nullptr ? colmap->from : rows;
        if ((rowmap != nullptr ? rowmap->to : from) != rows || (colmap != nullptr ? colmap->to : from) != cols ||
            (colmap != nullptr && colmap->from != from))
            fail("sparsity %s: pair %d does not map one set to rows %s and columns %s", name, k, rows->name,
                 cols->name);
        sp->pairs.push_back({rowmap, colmap, from, rowmap != nullptr ? rowmap->dim : 1,
                             colmap != nullptr ? colmap->dim : 1, {}});
    }
    auto identity = [](const op_sparsity_core::Pair& p) { return p.rowmap == nullptr && p.colmap == nullptr; };
    if (rows == cols && std::none_of(sp->pairs.begin(), sp->pairs.end(), identity))
        sp->pairs.push_back({nullptr, nullptr, rows, 1, 1, {}});
    build(sp, library->order());
    sparsities.push_back(sp);
    return sp;
}

op_mat op_decl_mat(op_sparsity sparsity, int dim, char const *type, char const *name, char const *format) {
    if (std::strcmp(type, "double") != 0)
        fail("matrix %s: type %s, only double is supported", name, type);

    op_mat A = new op_mat_core{};
    A->sparsity = sparsity;
    A->dim = dim;
    A->name = name;
    A->format = format != nullptr ? format : settings.format;
    A->layout.assign(sparsity->nnz->size, Entry{});
    mats.push_back(A);

    A->lib = library->matrix(A, A->format);
    finish_storage(A);
    std::size_t bytes = A->layout.size() * sizeof(Entry);
    for (const Region& r : A->regions)
        bytes += r.owned ? r.bytes : 0;
    report_memory(("matrix " + A->name + " storage").c_str(), bytes);
    return A;
}

op_arg op_arg_mat(op_mat A, int rowidx, op_map rowmap, int colidx, op_map colmap, int dim, char const *type,
                  op_access acc) {
    if (dim != A->dim || std::strcmp(type, "double") != 0)
        fail("op_arg_mat on %s: %d x %d %s blocks, the matrix has %d x %d double", A->name.c_str(), dim, dim,
             type, A->dim, A->dim);
    if (acc != OP_INC && acc != OP_READ)
        fail("op_arg_mat on %s: access must be OP_INC or OP_READ", A->name.c_str());

    for (auto& pair : A->sparsity->pairs) {
        if (pair.rowmap != rowmap || pair.colmap != colmap)
            continue;
        int i = rowmap != nullptr ? rowidx : 0, j = colmap != nullptr ? colidx : 0;
        if (i < 0 || i >= pair.rowdim || j < 0 || j >= pair.coldim)
            fail("op_arg_mat on %s: index (%d, %d) outside the maps", A->name.c_str(), rowidx, colidx);
        op_map slots = pair.slots[i];
        op_arg arg{};
        arg.index = -1;
        arg.map = slots;
        arg.dim = dim * dim;
        arg.idx = j;
        arg.size = dim * dim * (int)sizeof(double);
        arg.map_data = slots->map;
        arg.map_data_d = slots->map_d;
        arg.type = type;
        arg.acc = acc;
        arg.argtype = OP_ARG_MAT;
        arg.opt = 1;
        arg.mat = &A->view;
        return arg;
    }
    fail("op_arg_mat on %s: its sparsity has no pair (%s, %s)", A->name.c_str(),
         rowmap != nullptr ? rowmap->name : "OP_ID", colmap != nullptr ? colmap->name : "OP_ID");
}

void op_mat_zero(op_mat A) {
    for (const Region& r : A->regions)
        if (r.data != nullptr)
            zero(r.data, r.bytes);
    std::fill(A->untranslated.begin(), A->untranslated.end(), 0.0);
    A->pending = false;
}

void op_mat_assemble(op_mat A) {
    if (A->pending) {
        untranslated_pass<false>(A, A->untranslated.data());
        std::fill(A->untranslated.begin(), A->untranslated.end(), 0.0);
        A->pending = false;
    }
#if OP2_SOLVE_DEVICE
    GPU_CHECK(gpuDeviceSynchronize());
#endif
    A->lib->assemble();
    if (A->state++ == 0)
        report_memory(("matrix " + A->name + " assembled").c_str());
}

op_solver op_decl_solver(op_mat A, char const *name, char const *options) {
    op_sparsity sp = A->sparsity;
    if (sp->rows != sp->cols)
        fail("solver %s: matrix %s is not square", name, A->name.c_str());

    op_solver s = new op_solver_core{};
    s->A = A;
    s->name = name;
    s->options = parse_options(options);
    solvers.push_back(s);
    return s;
}

void op_solver_set_pc_mat(op_solver s, op_mat P) {
    if (P->sparsity->rows != s->A->sparsity->rows || P->dim != s->A->dim)
        fail("solver %s: preconditioning matrix %s does not match %s", s->name.c_str(), P->name.c_str(),
             s->A->name.c_str());
    s->P = P;
}

static void make_callback_dats(op_solver s) {
    if (s->in != nullptr)
        return;
    op_set rows = s->A->sparsity->rows;
    s->in = op_decl_dat_temp_char(rows, s->A->dim, "double", sizeof(double), (s->name + "_in").c_str());
    s->out = op_decl_dat_temp_char(rows, s->A->dim, "double", sizeof(double), (s->name + "_out").c_str());
}

void op_solver_set_pc(op_solver s, op_apply_fn apply, void *ctx) {
    s->pc = apply;
    s->pc_ctx = ctx;
    s->options.pc = "custom";
    make_callback_dats(s);
}

void op_solver_set_operator(op_solver s, op_apply_fn apply, void *ctx) {
    s->op = apply;
    s->op_ctx = ctx;
    make_callback_dats(s);
}

int op_solve(op_solver s, op_dat b, op_dat x) {
    op_set rows = s->A->sparsity->rows;
    if (b->set != rows || x->set != rows || b->dim != s->A->dim || x->dim != s->A->dim)
        fail("op_solve %s: b and x must be dim-%d dats on %s", s->name.c_str(), s->A->dim, rows->name);
    if ((OP_auto_soa && b->dim > 1) || std::strstr(b->type, ":soa") || std::strstr(x->type, ":soa"))
        fail("op_solve %s: SoA vectors are not supported", s->name.c_str());

    if (!s->A->lib || (s->P != nullptr && !s->P->lib))
        fail("op_solve %s: its matrices have not been assembled", s->name.c_str());
    if (!s->lib)
        s->lib = library->solver(s);
    op_arg args[] = {op_arg_dat_core(b, -1, OP_ID, b->dim, "double", OP_READ),
                     op_arg_dat_core(x, -1, OP_ID, x->dim, "double", OP_RW)};
    enter(rows, args, 2);
    bool converged = s->lib->solve(b, x);
    leave(args, 2);
    static bool first = true;
    if (first)
        report_memory(("solver " + s->name + " solved").c_str());
    first = false;
    return converged ? 1 : 0;
}

void op_solver_stats(op_solver s, int *iterations, double *residual) {
    if (iterations != nullptr)
        *iterations = s->lib ? s->lib->iterations : 0;
    if (residual != nullptr)
        *residual = s->lib ? s->lib->residual : 0.0;
}

} // extern "C"
