// Not intended to be used with OP_NO_REALLOC flag

#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"
#include "op_profile.h"

#include "../utility.h"

#include <cstdint>
#include <cstring>

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
inline void add3(double *g, const double *v) {
  for (int i = 0; i < 3; ++i)
    g[i] += v[i];
}
// A global of dimension 3 incremented in a helper, which indexes each
// thread's copy as the kernel would.
void inc3_helper(const double *dat, double *g) {
  add3(g, dat);
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

void opt_read5(double *dat, const double *g, const int *on) {
  for (int i = 0; i < 5; ++i)
    dat[i] = *on ? g[i] : -1.0;
}

void read_nonfinite(double *d, float *f, const double *gd, const float *gf) {
  *d = *gd;
  *f = *gf;
}

void opt_read_inc(const double *dat, const double *g, double *sum, const int *on) {
  if (*on)
    *sum += *dat + *g;
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);
  op_profile_start("CppGblArgTests");

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

  std::vector<double> data_nf_d(local_size, 0.0);
  std::vector<float> data_nf_f(local_size, 0.0f);
  op_dat dat_nf_d = op_decl_dat(set, 1, "double", data_nf_d.data(), "dat_nf_d");
  op_dat dat_nf_f = op_decl_dat(set, 1, "float", data_nf_f.data(), "dat_nf_f");

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
    double g_inc3[3] = {0.0, 0.0, 0.0};

    op_par_loop(inc3_helper, "inc3_helper", set,
              op_arg_dat(dat_iota5, -1, OP_ID, 5, "double", OP_READ),
              op_arg_gbl(g_inc3, 3, "double", OP_INC));

    double expected[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < gbl_size; ++i)
      for (int d = 0; d < 3; ++d)
        expected[d] += (double)(i * 5 + d);

    for (int d = 0; d < 3; ++d)
      check(std::abs(g_inc3[d] - expected[d]) < TOL, d, "inc3_helper failed");

    printf("inc3_helper passed\n");
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

  // --- READ of non-finite values, which a JIT kernel has as constants ---
  {
    double g_ninf = -std::numeric_limits<double>::infinity();
    const uint32_t nan_bits = 0x7fc00123u;
    float g_nan;
    std::memcpy(&g_nan, &nan_bits, sizeof(g_nan));

    // Enough calls for the loop to reach its JIT kernel.
    for (int pass = 0; pass < 12; ++pass)
      op_par_loop(read_nonfinite, "read_nonfinite", set,
                op_arg_dat(dat_nf_d, -1, OP_ID, 1, "double", OP_WRITE),
                op_arg_dat(dat_nf_f, -1, OP_ID, 1, "float", OP_WRITE),
                op_arg_gbl(&g_ninf, 1, "double", OP_READ),
                op_arg_gbl(&g_nan, 1, "float", OP_READ));

    std::vector<double> fetched_d(local_size);
    std::vector<float> fetched_f(local_size);
    op_fetch_data(dat_nf_d, fetched_d.data());
    op_fetch_data(dat_nf_f, fetched_f.data());
    for (int i = 0; i < set->size; ++i) {
      uint32_t bits;
      std::memcpy(&bits, &fetched_f[i], sizeof(bits));
      check(fetched_d[i] == g_ninf && bits == nan_bits, i, "read_nonfinite failed");
    }

    printf("read_nonfinite passed\n");
  }

  // --- Optional READ and INC, active then inactive; the inactive READ has no data ---
  {
    double g_read = 2.0;
    double g_sum = 0.0;

    for (int on = 1; on >= 0; --on) {
      double *g_read_ptr = on ? &g_read : nullptr;
      op_par_loop(opt_read_inc, "opt_read_inc", set,
                op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ),
                op_opt_arg_gbl(on, g_read_ptr, 1, "double", OP_READ),
                op_opt_arg_gbl(on, &g_sum, 1, "double", OP_INC),
                op_arg_gbl(&on, 1, "int", OP_READ));
    }

    double expected = (gbl_size - 1) * gbl_size / 2 + gbl_size * g_read;
    check(std::abs(g_sum - expected) < TOL, 0, "opt_read_inc failed");

    printf("opt_read_inc passed\n");
  }

  // --- READ of an array over enough passes for its JIT kernel, the value
  // changing near the end ---
  {
    double g[5] = {33.0, 43.0, 53.0, 63.0, 73.0};
    std::vector<double> fetched(local_size * 5, 0.0);
    for (int pass = 0; pass < 12; ++pass) {
      if (pass == 10)
        for (int d = 0; d < 5; ++d)
          g[d] += 1.0;
      op_par_loop(read5, "read5_passes", set,
                op_arg_dat(dat5, -1, OP_ID, 5, "double", OP_WRITE),
                op_arg_gbl(g, 5, "double", OP_READ));

      op_fetch_data(dat5, fetched.data());
      for (int i = 0; i < local_size; ++i)
        for (int d = 0; d < 5; ++d)
          check(std::abs(fetched[i * 5 + d] - g[d]) < TOL, i * 5 + d, "read5_passes failed");
    }

    printf("read5_passes passed\n");
  }

  // --- Optional READ of an array, inactive then active, each long enough
  // for its JIT kernel; the inactive one has no data ---
  {
    double g[5] = {3.0, 4.0, 5.0, 6.0, 7.0};
    std::vector<double> fetched(local_size * 5, 0.0);
    for (int pass = 0; pass < 20; ++pass) {
      int on = pass >= 10;
      op_par_loop(opt_read5, "opt_read5", set,
                op_arg_dat(dat5, -1, OP_ID, 5, "double", OP_WRITE),
                op_opt_arg_gbl(on, on ? g : nullptr, 5, "double", OP_READ),
                op_arg_gbl(&on, 1, "int", OP_READ));

      op_fetch_data(dat5, fetched.data());
      for (int i = 0; i < local_size; ++i)
        for (int d = 0; d < 5; ++d)
          check(std::abs(fetched[i * 5 + d] - (on ? g[d] : -1.0)) < TOL, i * 5 + d,
                "opt_read5 failed");
    }

    printf("opt_read5 passed\n");
  }

  op_profile_end();
  op_profile_output();

  op_exit();

  return 0;
}