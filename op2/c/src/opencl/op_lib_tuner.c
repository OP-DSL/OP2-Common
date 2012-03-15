#include "op_lib_tuner.h"

op_tuner  *OP_tuners;
op_tuner  *OP_global_tuner;

///////////////////
//// OP_TUNER construction
///////////////////

//external variables declared in op_lib_core.c
extern int OP_part_size, OP_block_size, OP_cache_line_size;

int OP_tuner_index = 0;

op_tuner* op_create_tuner(char const *name) {
  int it = 0;
  while(it < OP_tuner_index) {
    if((strcmp(name, OP_tuners[it].name) == 0) ) {
      return &OP_tuners[it];
      }
      ++it;
    }

  OP_tuners = (op_tuner*) realloc(OP_tuners, (OP_tuner_index+1)*sizeof(op_tuner));
  OP_tuners[OP_tuner_index].loop_tuner = 1;
  OP_tuners[OP_tuner_index].name = name;
  OP_tuners[OP_tuner_index].block_size = OP_global_tuner->block_size;
  OP_tuners[OP_tuner_index].part_size = OP_global_tuner->part_size;
  OP_tuners[OP_tuner_index].cache_line_size = OP_global_tuner->cache_line_size;
  OP_tuners[OP_tuner_index].op_warpsize = 1;
  OP_tuners[OP_tuner_index].architecture = ANY;
  OP_tuners[OP_tuner_index].active = 1;
  ++OP_tuner_index;
  return &OP_tuners[OP_tuner_index-1];
}

op_tuner* op_tuner_core(char const *name) {
  return NULL;
}

op_tuner* op_tuner_get(char const *name) {
  int it = 0;
  while(it < OP_tuner_index) {
    if((strcmp(name, OP_tuners[it].name) == 0)) {
      return &OP_tuners[it];
    }
  }
  return NULL;
}

op_tuner * op_create_global_tuner() {
  if(OP_global_tuner == NULL) {
    OP_global_tuner = (op_tuner*) malloc(sizeof(op_tuner));
  }

  OP_global_tuner->name = "Global tuner";
  OP_global_tuner->op_warpsize = 1;
  OP_global_tuner->part_size = 128;
  OP_global_tuner->block_size = 128;
  OP_global_tuner->cache_line_size = 128;
  OP_global_tuner->loop_tuner = 0;
  OP_global_tuner->active = 1;
  OP_global_tuner->architecture = ANY;

  return OP_global_tuner;
}

op_tuner* op_get_global_tuner(){
  return OP_global_tuner;
}

op_tuner * op_init_global_tuner(int argc, char **argv) {
  if(OP_global_tuner == NULL) {
    OP_global_tuner = op_create_global_tuner();
  }

#ifdef OP_BLOCK_SIZE
  OP_block_size = OP_BLOCK_SIZE;
  OP_global_tuner->block_size = OP_block_size;
#else
  OP_block_size = OP_global_tuner->block_size;
#endif

#ifdef OP_PART_SIZE
  OP_part_size = OP_PART_SIZE;
  OP_global_tuner->part_size = OP_part_size;
#else
  OP_part_size = OP_global_tuner->part_size;
#endif

#ifdef OP_CACHE_LINE_SIZE
  OP_cache_line_size = OP_CACHE_LINE_SIZE;
  OP_global_tuner->cache_line_size = OP_cache_line_size;
#else
  OP_cache_line_size = OP_global_tuner->cache_line_size;
#endif

  for (int n=1; n<argc; n++) {
    if (strncmp(argv[n],"OP_BLOCK_SIZE=",14)==0) {
      OP_block_size = atoi(argv[n]+14);
      OP_global_tuner->block_size = OP_block_size;
      printf("\n OP_block_size = %d \n", OP_block_size);
    }
    if (strncmp(argv[n],"OP_PART_SIZE=",13)==0) {
      OP_part_size = atoi(argv[n]+13);
      OP_global_tuner->part_size = OP_part_size;
      printf("\n OP_part_size  = %d \n", OP_part_size);
    }
    if (strncmp(argv[n],"OP_CACHE_LINE_SIZE=",19)==0) {
      OP_cache_line_size = atoi(argv[n]+19);
      OP_global_tuner->cache_line_size = OP_cache_line_size;
      printf("\n OP_cache_line_size  = %d \n", OP_cache_line_size);
    }
  }

  return OP_global_tuner;
}
