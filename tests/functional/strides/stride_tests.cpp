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
void write5(double *dat, const double *read) {
  double power = 1.0;
  for (int i = 0; i < 5; ++i) {
    dat[i] = (*read) * (i + 1) * power;
    power *= 10;
  }
}
void write5_within_kernel(double *dat0, double *dat1, const double *read) {
  double power = 1.0;
  for (int i = 0; i < 5; ++i) {
    dat0[i] = (*read) * (i + 1) * power;
    power *= 0.1;
  }
  write5(dat1, read);
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);

  int my_rank = 0;
  int comm_size = 1;

  get_rank_and_size(my_rank, comm_size);

  // Use a non-multiple of comm_size, 32 or 64 to make full use of these tests
  constexpr int gbl_size = 256; 

  const int local_size = compute_local_size(gbl_size, comm_size, my_rank);
  double local_start = (double)get_local_start(gbl_size, comm_size, my_rank);

  op_set set = op_decl_set(local_size, "my_set");
  printf("local_size = %d, set size = %d\n", local_size, set->size);

  std::vector<double> data5_0(local_size * 5, 0.0);
  std::vector<double> data5_1(local_size * 5, 0.0);
  std::vector<double> data1(local_size, 0.0);
  std::iota(data1.data(), data1.data() + local_size, local_start);

  op_dat dat5_0 = op_decl_dat(set, 5, "double", data5_0.data(), "dat5_0");
  op_dat dat5_1 = op_decl_dat(set, 5, "double", data5_1.data(), "dat5_1");
  op_dat dat_iota1 = op_decl_dat(set, 1, "double", data1.data(), "dat_iota1");

  op_partition("", "", NULL, NULL, NULL);
  
  // --- Regular Stride Tests ---
  {
    op_par_loop(write5, "write5", set,
              op_arg_dat(dat5_0, -1, OP_ID, 5, "double", OP_WRITE),
              op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ));
  
    std::vector<double> data_fetched5(local_size * 5, 0.0);
    op_fetch_data(dat5_0, data_fetched5.data());
    
    for (int i = 0; i < local_size; ++i) {
      for (int d = 0; d < 5; ++d) {
        double expected = (local_start + i) * (d + 1) * std::pow(10, d);
        check(std::abs(data_fetched5[i * 5 + d] - expected) < TOL, i * 5 + d, "WRITE5 failed");
      }
    }

    printf("write5 passed\n");
  }

  // --- Function call within kernel Stride Tests ---
  {
    op_par_loop(write5_within_kernel, "write5_within_kernel", set,
              op_arg_dat(dat5_0, -1, OP_ID, 5, "double", OP_WRITE),
              op_arg_dat(dat5_1, -1, OP_ID, 5, "double", OP_WRITE),
              op_arg_dat(dat_iota1, -1, OP_ID, 1, "double", OP_READ));
  
    std::vector<double> data_fetched5_0(local_size * 5, 0.0);
    op_fetch_data(dat5_0, data_fetched5_0.data());

    std::vector<double> data_fetched5_1(local_size * 5, 0.0);
    op_fetch_data(dat5_1, data_fetched5_1.data());

    for (int i = 0; i < local_size; ++i) {
      for (int d = 0; d < 5; ++d) {
        double expected_0 = (local_start + i) * (d + 1) / std::pow(10, d);
        double expected_1 = (local_start + i) * (d + 1) * std::pow(10, d);
        check(std::abs(data_fetched5_0[i * 5 + d] - expected_0) < TOL, i * 5 + d, "write5_within_kernel failed");
        check(std::abs(data_fetched5_1[i * 5 + d] - expected_1) < TOL, i * 5 + d, "write5_within_kernel failed");
      }
    }

    printf("write5_within_kernel passed\n");
  }

  op_exit();

  return 0;
}