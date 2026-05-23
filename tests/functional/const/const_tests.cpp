// Not intended to be used with OP_NO_REALLOC flag

#include "op_seq.h"
#include "op_profile.h"
#include <vector>

#define TOL 1e-9

// --- Utility functions ---
void check(bool cond, int idx, const char *msg) {
  if (!cond) {
    printf("ERROR: %s at idx: %d\n", msg, idx);
    op_exit();
    exit(EXIT_FAILURE);
  }
}

// --- CONSTANTS ---
double my_const1 = 20.1;
double my_const4[4] = {30.1, 40.2, 50.3, 60.4};

// --- KERNELS ---
void consts1(double *dat) {
  *dat = my_const1;
}
void consts4(double *dat) {
  for (int i = 0; i < 4; ++i)
    dat[i] = my_const4[i];
}

// --- main ---
int main(int argc, char **argv) {

  op_init(argc, argv, 2);
  op_profile_start("CppConstTests");

  constexpr int size = 32;
  op_set set = op_decl_set(size, "my_set");;
  printf("set size = %d\n", set->size);

  std::vector<double> data(size * 4, 0.0);

  op_dat dat1 = op_decl_dat(set, 1, "double", data.data(), "dat1");
  op_dat dat4 = op_decl_dat(set, 4, "double", data.data(), "dat4");

  op_decl_const(1, "double", &my_const1);
  op_decl_const(4, "double", my_const4);

  op_partition("", "", NULL, NULL, NULL);

  // --- CONST Check ---
  {
    op_par_loop(consts1, "consts1", set,
              op_arg_dat(dat1, -1, OP_ID, 1, "double", OP_WRITE));
  
    std::vector<double> data_fetched(size, 0.0);
    op_fetch_data(dat1, data_fetched.data());
    for (int i = 0; i < size; ++i) // No need to check for all elements
      check(std::abs(data_fetched[i] - my_const1) < TOL, i, "consts1 failed");
    printf("consts1 passed\n");
  }
  {
    op_par_loop(consts4, "consts4", set,
              op_arg_dat(dat4, -1, OP_ID, 4, "double", OP_WRITE));
  
    std::vector<double> data_fetched4(size * 4, 0.0);
    op_fetch_data(dat4, data_fetched4.data());
    for (int i = 0; i < size; ++i) // No need to check for all elements
      for (int d = 0; d < 4; ++d)
        check(std::abs(data_fetched4[i * 4 + d] - my_const4[d]) < TOL, i * 4 + d, "consts4 failed");
    printf("consts4 passed\n");
  }

  op_profile_end();
  op_profile_output();

  op_exit();

  return 0;
}