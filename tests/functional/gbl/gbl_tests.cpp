// Not intended to be used with OP_NO_REALLOC flag

#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"

#include "../utility.h"

#define TOL 1e-9

// --- Utility functions ---
void check(bool cond, int idx, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d\n", msg, idx);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

// --- KERNELS ---
void read1(double *dat, const double *g) {
  *dat = *g;
}
void read5(double *dat, const double *g) {
  for (int i = 0; i < 5; ++i)
    dat[i] = g[i];
}

void inc1(const double *dat, double *g) {
  *g += *dat;
}
void inc5(const double *dat, double *g) {
  for (int i = 0; i < 5; ++i)
    g[i] += dat[i];
}

void min1(const double *dat, double *g) {
  *g = MIN(*g, *dat);
}
void min5(const double *dat, double *g) {
  for (int i = 0; i < 5; ++i)
    g[i] = MIN(g[i], dat[i]);
}

void max1(const double *dat, double *g) {
  *g = MAX(*g, *dat);
}
void max5(const double *dat, double *g) {
  for (int i = 0; i < 5; ++i)
    g[i] = MAX(g[i], dat[i]);
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);

  int my_rank = 0;
  int comm_size = 1;

  get_rank_and_size(my_rank, comm_size);

  // Use a non-multiple of comm_size, 32 or 64 to make full use of these tests
  constexpr int gbl_size = 161; 

  const int local_size = compute_local_size(gbl_size, comm_size, my_rank);
  double local_start = (double)get_local_start(gbl_size, comm_size, my_rank);

  op_set set = op_decl_set(local_size, "my_set");;
  printf("set size = %d\n", set->size);

  std::vector<double> data1(local_size, 0.0);
  std::vector<double> data5(local_size * 5, 0.0);

  op_dat dat1 = op_decl_dat(set, 1, "double", data1.data(), "dat1");
  op_dat dat5 = op_decl_dat(set, 5, "double", data5.data(), "dat5");

  std::iota(data1.data(), data1.data() + local_size, local_start);
  std::iota(data5.data(), data5.data() + local_size * 5, local_start * 5);

  op_dat dat_iota1 = op_decl_dat(set, 1, "double", data1.data(), "dat_iota1");
  op_dat dat_iota5 = op_decl_dat(set, 5, "double", data5.data(), "dat_iota5");

  op_partition("", "", NULL, NULL, NULL);
  
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
    double g_read5[5] = {30.0, 40.0, 50.0, 60.0, 70.0};
    op_par_loop(read5, "read5", set,
              op_arg_dat(dat5, -1, OP_ID, 5, "double", OP_WRITE),
              op_arg_gbl(g_read5, 5, "double", OP_READ));
  
    std::vector<double> data_fetched5(local_size * 5, 0.0);
    op_fetch_data(dat5, data_fetched5.data());
    
    for (int i = 0; i < local_size; ++i) {
      for (int d = 0; d < 5; ++d) {
        check(std::abs(data_fetched5[i * 5 + d] - g_read5[d]) < TOL, i * 5 + d, "READ5 failed");
      }
    }

    printf("read5 passed\n");
  }

  // --- INC ---
  {
    double g_inc1 = 0.0;

    op_par_loop(inc1, "inc1", set,
              op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ),
              op_arg_gbl(&g_inc1, 1, "double", OP_INC));
  
    double expected = (gbl_size - 1) * gbl_size / 2;
    // printf("inc1 set size: %d expected: %lf g_inc1: %lf\n", set->size, expected, g_inc1);
    check(std::abs(g_inc1 - expected) < TOL, 0, "inc1 dat failed");

    printf("inc1 passed\n");
  }
  {
    double g_inc5[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    op_par_loop(inc5, "inc5", set,
              op_arg_dat(dat_iota5, -1, OP_ID, 5, "double", OP_READ),
              op_arg_gbl(g_inc5, 5, "double", OP_INC));
  
    double expected[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < gbl_size; ++i)
      for (int d = 0; d < 5; ++d)
        expected[d] += (double)(i * 5 + d);

    for (int d = 0; d < 5; ++d) {
      // printf("expected: %lf g_inc5: %lf\n", expected[d], g_inc5[d]);
      check(std::abs(g_inc5[d] - expected[d]) < TOL, d, "inc5 dat failed");
    }

    printf("inc5 passed\n");
  }

  // --- MIN ---
  {
    double g_min1 = std::numeric_limits<double>::max();

    op_par_loop(min1, "min1", set,
                op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ),
                op_arg_gbl(&g_min1, 1, "double", OP_MIN));

    double expected = 0.0;
    // printf("min1 set size: %d expected: %lf g_min1: %lf\n", set->size, expected, g_min1);
    check(std::abs(g_min1 - expected) < TOL, 0, "min1 failed");
    
    printf("min1 passed\n");
  }
  {
    double g_min5[5] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 
                        0.4, std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};

    op_par_loop(min5, "min5", set,
                op_arg_dat(dat_iota5, -1, OP_ID, 5, "double", OP_READ),
                op_arg_gbl(g_min5, 5, "double", OP_MIN));

    double expected[5] = {0.0, 1.0, 0.4, 3.0, 4.0};

    for (int d = 0; d < 5; ++d) {
      // printf("min5 set size: %d expected: %lf g_min5: %lf\n", set->size, expected[d], g_min5[d]);
      check(std::abs(g_min5[d] - expected[d]) < TOL, d, "min5 failed");
    }

    printf("min5 passed\n");
  }

  // --- MAX ---
  {
    double g_max1 = std::numeric_limits<double>::min();

    op_par_loop(max1, "max1", set,
                op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ),
                op_arg_gbl(&g_max1, 1, "double", OP_MAX));

    double expected = (gbl_size - 1);
    // printf("max1 set size: %d expected: %lf g_max1: %lf\n", set->size, expected, g_max1);
    check(std::abs(g_max1 - expected) < TOL, 0, "max1 failed");
    
    printf("max1 passed\n");
  }
  {
    double g_max5[5] = {std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), 
                        1000000000.4, std::numeric_limits<double>::min(), std::numeric_limits<double>::min()};

    op_par_loop(max5, "max5", set,
                op_arg_dat(dat_iota5, -1, OP_ID, 5, "double", OP_READ),
                op_arg_gbl(g_max5, 5, "double", OP_MAX));

    double expected[5] = {gbl_size * 5 - 5, gbl_size * 5 - 4, 1000000000.4, gbl_size * 5 - 2, gbl_size * 5 - 1};

    for (int d = 0; d < 5; ++d) {
      // printf("max5 set size: %d expected: %lf g_max5: %lf\n", set->size, expected[d], g_max5[d]);
      check(std::abs(g_max5[d] - expected[d]) < TOL, d, "max5 failed");
    }

    printf("max5 passed\n");
  }

  op_exit();

  return 0;
}