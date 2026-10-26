// Not intended to be used with OP_NO_REALLOC flag

#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"
#include "op_profile.h"

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

inline void weigh3(const double *v, double *out) {
  *out = v[0] + 10.0 * v[1] + 100.0 * v[2];
}
// One helper called with a dat on the loop's set, one on another set (each
// with its own stride under SoA) and a local array.
void weigh_shared(double *r, const double *own, const double *other) {
  const double local[3] = {1.0, 2.0, 3.0};
  double a, b, c;
  weigh3(own, &a);
  weigh3(other, &b);
  weigh3(local, &c);
  r[0] = a + 2.0 * b + 4.0 * c;
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);
  op_profile_start("CppStrideTests");

  int my_rank = 0;
  int comm_size = 1;

  get_rank_and_size(my_rank, comm_size);

  // Use a non-multiple of comm_size, 32 or 64 to make full use of these tests
  constexpr int gbl_size = 201; 

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

  // A second set of another size, reached through a map.
  const int other_size = local_size + 17;
  op_set other = op_decl_set(other_size, "other_set");
  std::vector<int> to_other(local_size);
  for (int i = 0; i < local_size; ++i)
    to_other[i] = (7 * i) % other_size;
  op_map set_to_other = op_decl_map(set, other, 1, to_other.data(), "set_to_other");

  std::vector<double> own3(local_size * 3), other3(other_size * 3), weighed(local_size, 0.0);
  for (int i = 0; i < local_size * 3; ++i)
    own3[i] = local_start * 3 + i;
  for (int j = 0; j < other_size * 3; ++j)
    other3[j] = 1000.0 + j;
  op_dat dat_own3 = op_decl_dat(set, 3, "double", own3.data(), "dat_own3");
  op_dat dat_other3 = op_decl_dat(other, 3, "double", other3.data(), "dat_other3");
  op_dat dat_weighed = op_decl_dat(set, 1, "double", weighed.data(), "dat_weighed");

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

  // --- A helper shared by dats of two sets and a local ---
  {
    op_par_loop(weigh_shared, "weigh_shared", set,
              op_arg_dat(dat_weighed, -1, OP_ID, 1, "double", OP_WRITE),
              op_arg_dat(dat_own3, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(dat_other3, 0, set_to_other, 3, "double", OP_READ));

    std::vector<double> fetched(local_size, 0.0);
    op_fetch_data(dat_weighed, fetched.data());

    auto weigh = [](const double *v) { return v[0] + 10.0 * v[1] + 100.0 * v[2]; };
    const double local[3] = {1.0, 2.0, 3.0};
    for (int i = 0; i < local_size; ++i) {
      double expected = weigh(&own3[i * 3]) + 2.0 * weigh(&other3[to_other[i] * 3]) + 4.0 * weigh(local);
      check(std::abs(fetched[i] - expected) < TOL, i, "weigh_shared failed");
    }

    printf("weigh_shared passed\n");
  }

  op_profile_end();
  op_profile_output();

  op_exit();

  return 0;
}