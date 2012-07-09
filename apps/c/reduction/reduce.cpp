#include <stdio.h>
#include <stdlib.h>
#include "op_seq.h"
#include "count.h"
#include "addto.h"
#include "addto_id.h"

int main(int argc, char **argv)
{

  int n = 10;
  op_init(argc, argv, 5);
  op_set set = op_decl_set(n, "set");

  op_set set2 = op_decl_set(n, "set");

  int *map_d;
  map_d = (int *)malloc(n * sizeof(int));
  for ( int i = 0; i < n; i++ ) {
    map_d[i] = n - i - 1;
  }
  op_map map = op_decl_map(set, set2, 1, map_d, "map");

  int *dat_d;
  dat_d = (int *)malloc(n * sizeof(int));
  for ( int i = 0; i < n; i++ ) {
    dat_d[i] = i+1;
  }
  op_dat dat = op_decl_dat(set2, 1, "int", dat_d, "dat");

  int count_d = 0;

  op_par_loop(count, "count", set,
              op_arg_gbl(&count_d, 1, "int", OP_INC));

  printf("Count is %d, should be %d\n", count_d, n);

  count_d = 0;
  op_par_loop(addto, "addto", set2,
              op_arg_gbl(&count_d, 1, "int", OP_INC),
              op_arg_dat(dat, -1, OP_ID, 1, "int", OP_READ));

  printf("Count is %d, should be %d\n", count_d, n*(n+1)/2);

  count_d = 0;
  op_par_loop(addto_id, "addto_id", set,
              op_arg_gbl(&count_d, 1, "int", OP_INC),
              op_arg_dat(dat, 0, map, 1, "int", OP_READ));

  printf("Count is %d, should be %d\n", count_d, n*(n+1)/2);

  free(dat_d);
  free(map_d);
  op_exit();
}
