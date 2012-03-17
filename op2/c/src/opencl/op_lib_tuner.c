#include "op_lib_tuner.h"

//OP_tuners is represented by the following linked list
node  *head, *current;
op_tuner  *OP_global_tuner;

///////////////////
//// OP_TUNER construction
///////////////////

//external variables declared in op_lib_core.c
extern int OP_part_size, OP_block_size, OP_cache_line_size;

int OP_tuner_index = 0;

void add_node(op_tuner* tuner) {
  if(head == NULL) {
    head = (node *) malloc(sizeof(node));
    head->OP_tuner = tuner;
    head->next = NULL;
    current = head;
  } else {
    node* new_elem = (node *) malloc(sizeof(node));
    new_elem->OP_tuner = tuner;
    new_elem->next = NULL;
    current->next = new_elem;
    current = new_elem;
  }
}

op_tuner* get_node(char const *name) {
  node *elem = head;
  while(elem != NULL) {
    if(strcmp(elem->OP_tuner->name, name) == 0) {
      return elem->OP_tuner;
    }
    elem = elem->next;
  }
  return NULL;
}

op_tuner* op_create_tuner(char const *name) {
  op_tuner *elem = get_node(name);

  if(elem != NULL) {
    return elem;
  }

  op_tuner *tuner = (op_tuner*) malloc(sizeof(op_tuner));
  tuner->loop_tuner = 1;
  tuner->name = name;
  tuner->block_size = OP_global_tuner->block_size;
  tuner->part_size = OP_global_tuner->part_size;
  tuner->cache_line_size = OP_global_tuner->cache_line_size;
  tuner->op_warpsize = 1;
  tuner->architecture = ANY;
  tuner->active = 1;

  add_node(tuner);

/*  for(int a = 0; a < OP_tuner_index; ++a) {
    if(&OP_tuners[a] != NULL) {
      printf("OP_tuners[%d] %d name %s, block_size %d, part_size %d\n", a, &OP_tuners[a], OP_tuners[a].name, OP_tuners[a].block_size, OP_tuners[a].part_size);
    } else {
      printf("%d is null\n", a);
    }
  }*/

  return tuner;
}

op_tuner* op_tuner_core(char const *name) {
  return NULL;
}

op_tuner* op_tuner_get(char const *name) {
  return get_node(name);
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
