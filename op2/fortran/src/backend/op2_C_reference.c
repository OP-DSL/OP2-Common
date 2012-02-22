
/*
 * This file implements the reference (sequential) version of op_par_loop calls
 * to be used in Fortran
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../../include/op2_C_reference.h"


static void arg_set ( int displacement, /* set element */
                      op_dat arg,
                      int itemSel, /* map field to be used */
                      op_map mapIn,
                      char ** p_arg )
{
  int n2;

  if ( mapIn->dim == -1 ) /* global variable, no mapping at all */
  {
  n2 = 0;
  }

  if ( mapIn->dim == 0 ) /* identity mapping */
  {
    n2 = displacement;
  }
  if ( mapIn->dim > 0 ) /* standard pointers */
  {
    n2 = mapIn->map[itemSel + displacement * mapIn->dim];
  }

  *p_arg = (char *) ( arg->data + n2 * arg->size );
}

void op_par_loop_2 ( void (*subroutineName)(char *, char *), op_set set,
                     op_dat dat0, int itemSel0, op_map map0, op_access access0,
                     op_dat dat1, int itemSel1, op_map map1, op_access access1
                   )
{
  (void)access0;
  (void)access1;

  int i;

  for ( i = 0; i < set->size; i++ ) {
    char * ptr0, * ptr1;

    arg_set ( i, dat0, itemSel0, map0, &ptr0 );
    arg_set ( i, dat1, itemSel1, map1, &ptr1 );

    (*subroutineName) ( ptr0, ptr1 );
  }
}

void op_par_loop_5 ( void (*subroutineName)(char *, char *, char *, char *, char *), op_set set,
                     op_dat dat0, int itemSel0, op_map map0, op_access access0,
                     op_dat dat1, int itemSel1, op_map map1, op_access access1,
                     op_dat dat2, int itemSel2, op_map map2, op_access access2,
                     op_dat dat3, int itemSel3, op_map map3, op_access access3,
                     op_dat dat4, int itemSel4, op_map map4, op_access access4
                   )
{
  (void)access0;
  (void)access1;
  (void)access2;
  (void)access3;
  (void)access4;

  int i;

  for ( i = 0; i < set->size; i++ ) {

    char * ptr0, * ptr1, * ptr2, * ptr3, * ptr4;

    arg_set ( i, dat0, itemSel0, map0, &ptr0 );
    arg_set ( i, dat1, itemSel1, map1, &ptr1 );
    arg_set ( i, dat2, itemSel2, map2, &ptr2 );
    arg_set ( i, dat3, itemSel3, map3, &ptr3 );
    arg_set ( i, dat4, itemSel4, map4, &ptr4 );


    (*subroutineName) ( ptr0, ptr1, ptr2, ptr3, ptr4 );
  }
}

void op_par_loop_6 ( void (*subroutineName)(char *, char *, char *, char *, char *, char *), op_set set,
                     op_dat dat0, int itemSel0, op_map map0, op_access access0,
                     op_dat dat1, int itemSel1, op_map map1, op_access access1,
                     op_dat dat2, int itemSel2, op_map map2, op_access access2,
                     op_dat dat3, int itemSel3, op_map map3, op_access access3,
                     op_dat dat4, int itemSel4, op_map map4, op_access access4,
                     op_dat dat5, int itemSel5, op_map map5, op_access access5
                   )
{
  (void)access0;
  (void)access1;
  (void)access2;
  (void)access3;
  (void)access4;
  (void)access5;

  int i;

  for ( i = 0; i < set->size; i++ ) {

    char * ptr0, * ptr1, * ptr2, * ptr3, * ptr4, * ptr5;

    arg_set ( i, dat0, itemSel0, map0, &ptr0 );
    arg_set ( i, dat1, itemSel1, map1, &ptr1 );
    arg_set ( i, dat2, itemSel2, map2, &ptr2 );
    arg_set ( i, dat3, itemSel3, map3, &ptr3 );
    arg_set ( i, dat4, itemSel4, map4, &ptr4 );
    arg_set ( i, dat5, itemSel5, map5, &ptr5 );


    (*subroutineName) ( ptr0, ptr1, ptr2, ptr3, ptr4, ptr5 );

  }
}

void op_par_loop_8 ( void (*subroutineName)(char *, char *, char *, char *, char *, char *, char *, char *), op_set set,
                     op_dat dat0, int itemSel0, op_map map0, op_access access0,
                     op_dat dat1, int itemSel1, op_map map1, op_access access1,
                     op_dat dat2, int itemSel2, op_map map2, op_access access2,
                     op_dat dat3, int itemSel3, op_map map3, op_access access3,
                     op_dat dat4, int itemSel4, op_map map4, op_access access4,
                     op_dat dat5, int itemSel5, op_map map5, op_access access5,
                     op_dat dat6, int itemSel6, op_map map6, op_access access6,
                     op_dat dat7, int itemSel7, op_map map7, op_access access7
                   )
{
  (void)access0;
  (void)access1;
  (void)access2;
  (void)access3;
  (void)access4;
  (void)access5;
  (void)access6;
  (void)access7;

  int i;

  for ( i = 0; i < set->size; i++ ) {

    char * ptr0, * ptr1, * ptr2, * ptr3, * ptr4, * ptr5, * ptr6, * ptr7;

    arg_set ( i, dat0, itemSel0, map0, &ptr0 );
    arg_set ( i, dat1, itemSel1, map1, &ptr1 );
    arg_set ( i, dat2, itemSel2, map2, &ptr2 );
    arg_set ( i, dat3, itemSel3, map3, &ptr3 );
    arg_set ( i, dat4, itemSel4, map4, &ptr4 );
    arg_set ( i, dat5, itemSel5, map5, &ptr5 );
    arg_set ( i, dat6, itemSel6, map6, &ptr6 );
    arg_set ( i, dat7, itemSel7, map7, &ptr7 );


    (*subroutineName) ( ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7 );

  }
}

void op_par_loop_12 ( void (*subroutineName)(char *, char *, char *, char *, char *, char *, char *, char *, char *, char *, char *, char *), op_set set,
                      op_dat dat0, int itemSel0, op_map map0, op_access access0,
                      op_dat dat1, int itemSel1, op_map map1, op_access access1,
                      op_dat dat2, int itemSel2, op_map map2, op_access access2,
                      op_dat dat3, int itemSel3, op_map map3, op_access access3,
                      op_dat dat4, int itemSel4, op_map map4, op_access access4,
                      op_dat dat5, int itemSel5, op_map map5, op_access access5,
                      op_dat dat6, int itemSel6, op_map map6, op_access access6,
                      op_dat dat7, int itemSel7, op_map map7, op_access access7,
                      op_dat dat8, int itemSel8, op_map map8, op_access access8,
                      op_dat dat9, int itemSel9, op_map map9, op_access access9,
                      op_dat dat10, int itemSel10, op_map map10, op_access access10,
                      op_dat dat11, int itemSel11, op_map map11, op_access access11
                    )
{
  (void)access0;
  (void)access1;
  (void)access2;
  (void)access3;
  (void)access4;
  (void)access5;
  (void)access6;
  (void)access7;
  (void)access8;
  (void)access9;
  (void)access10;
  (void)access11;

  int i;

  for ( i = 0; i < set->size; i++ ) {

    char * ptr0, * ptr1, * ptr2, * ptr3, * ptr4, * ptr5, * ptr6, * ptr7, * ptr8, * ptr9, * ptr10, * ptr11;

    arg_set ( i, dat0, itemSel0, map0, &ptr0 );
    arg_set ( i, dat1, itemSel1, map1, &ptr1 );
    arg_set ( i, dat2, itemSel2, map2, &ptr2 );
    arg_set ( i, dat3, itemSel3, map3, &ptr3 );
    arg_set ( i, dat4, itemSel4, map4, &ptr4 );
    arg_set ( i, dat5, itemSel5, map5, &ptr5 );
    arg_set ( i, dat6, itemSel6, map6, &ptr6 );
    arg_set ( i, dat7, itemSel7, map7, &ptr7 );
    arg_set ( i, dat8, itemSel8, map8, &ptr8 );
    arg_set ( i, dat9, itemSel9, map9, &ptr9 );
    arg_set ( i, dat10, itemSel10, map10, &ptr10 );
    arg_set ( i, dat11, itemSel11, map11, &ptr11 );


    (*subroutineName) ( ptr0, ptr1, ptr2, ptr3, ptr4, ptr5, ptr6, ptr7, ptr8, ptr9, ptr10, ptr11 );

  }
}

