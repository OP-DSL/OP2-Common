#ifdef USE_MPI
#include <mpi.h>
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"

#include <vector>
#include <numeric>
#include <limits>

#define TOL 1e-9

// --- Utility functions ---
void check(bool cond, int idx, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d\n", msg, idx);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

int compute_local_size(int global_size, int mpi_comm_size, int mpi_rank) {
  const int base = global_size / mpi_comm_size;
  const int remainder = global_size % mpi_comm_size;
  return base + ((mpi_rank < remainder) ? 1 : 0);
}

int get_local_start(int global_size, int mpi_comm_size, int mpi_rank) {
  const int base = global_size / mpi_comm_size;
  const int remainder = global_size % mpi_comm_size;
  return mpi_rank * base + std::min(mpi_rank, remainder);
}

// --- KERNELS ---
void read1(double *dat, const double *g) {
  *dat = *g;
}
void read4(double *dat, const double *g) {
  for (int i = 0; i < 4; ++i)
    dat[i] = g[i];
}

void inc1(const double *dat, double *g) {
  *g += *dat;
}
void inc4(const double *dat, double *g) {
  for (int i = 0; i < 4; ++i)
    g[i] += dat[i];
}

void min1(const double *dat, double *g) {
  *g = MIN(*g, *dat);
}
void min4(const double *dat, double *g) {
  for (int i = 0; i < 4; ++i)
    g[i] = MIN(g[i], dat[i]);
}

void max1(const double *dat, double *g) {
  *g = MAX(*g, *dat);
}
void max4(const double *dat, double *g) {
  for (int i = 0; i < 4; ++i)
    g[i] = MAX(g[i], dat[i]);
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);

  int my_rank = 0;
  int comm_size = 1;

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  printf("MPI rank %d of %d\n", my_rank, comm_size);
#endif

  constexpr int gbl_size = 32;
  const int local_size = compute_local_size(gbl_size, comm_size, my_rank);
  double local_start = (double)get_local_start(gbl_size, comm_size, my_rank);

  op_set set = op_decl_set(local_size, "my_set");;
  printf("set size = %d\n", set->size);

  std::vector<double> data1(local_size, 0.0);
  std::vector<double> data4(local_size * 4, 0.0);

  op_dat dat1 = op_decl_dat(set, 1, "double", data1.data(), "dat1");
  op_dat dat4 = op_decl_dat(set, 4, "double", data4.data(), "dat4");

  std::iota(data1.data(), data1.data() + local_size, local_start);
  std::iota(data4.data(), data4.data() + local_size * 4, local_start * 4);

  op_dat dat_iota1 = op_decl_dat(set, 1, "double", data1.data(), "dat_iota1");
  op_dat dat_iota4 = op_decl_dat(set, 4, "double", data4.data(), "dat_iota4");

  op_partition("PARMETIS", "GEOM", set, NULL, dat4);
  
  // --- READ ---
  {
    double g_read1 = 20.0;
    op_par_loop(read1, "read1", set,
              op_arg_dat(dat1, -1, OP_ID, 1, "double", OP_WRITE),
              op_arg_gbl(&g_read1, 1, "double", OP_READ));
  
    std::vector<double> data_fetched(local_size, 0.0);
    op_fetch_data(dat1, data_fetched.data());
    
    for (int i = 0; i < local_size; ++i) {
      check(std::abs(data_fetched[i] - g_read1) < TOL, i, "read1 failed");
    }
    
    printf("read1 passed\n");
  }
  {
    double g_read4[4] = {30.0, 40.0, 50.0, 60.0};
    op_par_loop(read4, "read4", set,
              op_arg_dat(dat4, -1, OP_ID, 4, "double", OP_WRITE),
              op_arg_gbl(g_read4, 4, "double", OP_READ));
  
    std::vector<double> data_fetched4(local_size * 4, 0.0);
    op_fetch_data(dat4, data_fetched4.data());
    
    for (int i = 0; i < local_size; ++i) {
      for (int d = 0; d < 4; ++d) {
        check(std::abs(data_fetched4[i * 4 + d] - g_read4[d]) < TOL, i * 4 + d, "READ4 failed");
      }
    }

    printf("read4 passed\n");
  }

  // --- INC ---
  {
    double g_inc1 = 0.0;

    op_par_loop(inc1, "inc1", set,
              op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ),
              op_arg_gbl(&g_inc1, 1, "double", OP_INC));
  
    double expected = (gbl_size - 1) * gbl_size / 2;

    check(std::abs(g_inc1 - expected) < TOL, 0, "inc1 dat failed");

    printf("inc1 passed\n");
  }
  {
    double g_inc4[4] = {0.0, 0.0, 0.0, 0.0};

    op_par_loop(inc4, "inc4", set,
              op_arg_dat(dat_iota4, -1, OP_ID, 4, "double", OP_READ),
              op_arg_gbl(g_inc4, 4, "double", OP_INC));
  
    double expected[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < gbl_size; ++i)
      for (int d = 0; d < 4; ++d)
        expected[d] += (double)(i * 4 + d);

    for (int d = 0; d < 4; ++d) {
      check(std::abs(g_inc4[d] - expected[d]) < TOL, d, "inc4 dat failed");
    }

    printf("inc4 passed\n");
  }

  // --- MIN ---
  {
    double g_min1 = std::numeric_limits<double>::max();

    op_par_loop(min1, "min1", set,
                op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ),
                op_arg_gbl(&g_min1, 1, "double", OP_MIN));

    double expected = 0.0;

    check(std::abs(g_min1 - expected) < TOL, 0, "min1 failed");
    
    printf("min1 passed\n");
  }
  {
    double g_min4[4] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 
                        0.4, std::numeric_limits<double>::max()};

    op_par_loop(min4, "min4", set,
                op_arg_dat(dat_iota4, -1, OP_ID, 4, "double", OP_READ),
                op_arg_gbl(g_min4, 4, "double", OP_MIN));

    double expected[4] = {0.0, 1.0, 0.4, 3.0};

    for (int d = 0; d < 4; ++d) {
      check(std::abs(g_min4[d] - expected[d]) < TOL, d, "min4 failed");
    }

    printf("min4 passed\n");
  }

  // --- MAX ---
  {
    double g_max1 = std::numeric_limits<double>::min();

    op_par_loop(max1, "max1", set,
                op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ),
                op_arg_gbl(&g_max1, 1, "double", OP_MAX));

    double expected = (gbl_size - 1);

    check(std::abs(g_max1 - expected) < TOL, 0, "max1 failed");
    
    printf("max1 passed\n");
  }
  {
    double g_max4[4] = {std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), 
                        1000.4, std::numeric_limits<double>::min()};

    op_par_loop(max4, "max4", set,
                op_arg_dat(dat_iota4, -1, OP_ID, 4, "double", OP_READ),
                op_arg_gbl(g_max4, 4, "double", OP_MAX));

    double expected[4] = {gbl_size * 4 - 4, gbl_size * 4 - 3, 1000.4, gbl_size *  4 - 1};

    for (int d = 0; d < 4; ++d) {
      check(std::abs(g_max4[d] - expected[d]) < TOL, d, "max4 failed");
    }

    printf("max4 passed\n");
  }

  op_exit();

  return 0;
}