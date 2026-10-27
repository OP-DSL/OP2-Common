// A helper called with a dat on the loop's set, one on another set (each
// with its own stride under SoA) and a local array.

#ifdef USE_MPI
#include "op_lib_mpi.h"
#endif

#include "op_seq.h"
#include "op_profile.h"

#include "../utility.h"

#define TOL 1e-9

inline void weigh3(const double *v, double *out) {
  *out = v[0] + 10.0 * v[1] + 100.0 * v[2];
}

void weigh_shared(double *r, const double *own, const double *other) {
  const double local[3] = {1.0, 2.0, 3.0};
  double a, b, c;
  weigh3(own, &a);
  weigh3(other, &b);
  weigh3(local, &c);
  r[0] = a + 2.0 * b + 4.0 * c;
}

int main(int argc, char **argv) {
  op_init(argc, argv, 2);
  op_profile_start("CppSharedHelperTests");

  const int size = 201;
  const int other_size = size + 17;
  op_set set = op_decl_set(size, "my_set");
  op_set other = op_decl_set(other_size, "other_set");
  std::vector<int> to_other(size);
  for (int i = 0; i < size; ++i)
    to_other[i] = (7 * i) % other_size;
  op_map set_to_other = op_decl_map(set, other, 1, to_other.data(), "set_to_other");

  std::vector<double> own3(size * 3), other3(other_size * 3), weighed(size, 0.0);
  for (int i = 0; i < size * 3; ++i)
    own3[i] = i;
  for (int j = 0; j < other_size * 3; ++j)
    other3[j] = 1000.0 + j;
  op_dat dat_own3 = op_decl_dat(set, 3, "double", own3.data(), "dat_own3");
  op_dat dat_other3 = op_decl_dat(other, 3, "double", other3.data(), "dat_other3");
  op_dat dat_weighed = op_decl_dat(set, 1, "double", weighed.data(), "dat_weighed");

  op_partition("", "", NULL, NULL, NULL);

  op_par_loop(weigh_shared, "weigh_shared", set,
              op_arg_dat(dat_weighed, -1, OP_ID, 1, "double", OP_WRITE),
              op_arg_dat(dat_own3, -1, OP_ID, 3, "double", OP_READ),
              op_arg_dat(dat_other3, 0, set_to_other, 3, "double", OP_READ));

  std::vector<double> fetched(size, 0.0);
  op_fetch_data(dat_weighed, fetched.data());

  auto weigh = [](const double *v) { return v[0] + 10.0 * v[1] + 100.0 * v[2]; };
  const double local[3] = {1.0, 2.0, 3.0};
  for (int i = 0; i < size; ++i) {
    double expected = weigh(&own3[i * 3]) + 2.0 * weigh(&other3[to_other[i] * 3]) + 4.0 * weigh(local);
    if (std::abs(fetched[i] - expected) >= TOL) {
      printf("ERROR: weigh_shared failed at idx: %d\n", i);
      op_exit();
      exit(EXIT_FAILURE);
    }
  }
  printf("weigh_shared passed\n");

  op_profile_end();
  op_exit();
  return 0;
}
