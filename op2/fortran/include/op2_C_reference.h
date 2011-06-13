
#ifndef __OP2_C_REFERENCE_H
#define __OP2_C_REFERENCE_H

/*
 * This file declares the C functions used to implement the Fortran OP2 reference library
 */

#include <op_lib_core.h>

void arg_set ( int displacement, /* set element */
			   op_dat arg,
			   int itemSel, /* map field to be used */
			   op_map mapIn,
			   char ** p_arg );

void op_par_loop_2 ( void (*subroutineName)(char *, char *), op_set set,
					 op_dat dat0, int itemSel0, op_map map0, op_access access0,
					 op_dat dat1, int itemSel1, op_map map1, op_access access1
				   );

void op_par_loop_5 ( void (*subroutineName)(char *, char *, char *, char *, char *), op_set set,
					op_dat dat0, int itemSel0, op_map map0, op_access access0,
					op_dat dat1, int itemSel1, op_map map1, op_access access1,
					op_dat dat2, int itemSel2, op_map map2, op_access access2,
					op_dat dat3, int itemSel3, op_map map3, op_access access3,
					op_dat dat4, int itemSel4, op_map map4, op_access access4
				 );

void op_par_loop_6 ( void (*subroutineName)(char *, char *, char *, char *, char *, char *), op_set set,
					op_dat dat0, int itemSel0, op_map map0, op_access access0,
					op_dat dat1, int itemSel1, op_map map1, op_access access1,
					op_dat dat2, int itemSel2, op_map map2, op_access access2,
					op_dat dat3, int itemSel3, op_map map3, op_access access3,
					op_dat dat4, int itemSel4, op_map map4, op_access access4,
					op_dat dat5, int itemSel5, op_map map5, op_access access5
					);

void op_par_loop_8 ( void (*subroutineName)(char *, char *, char *, char *, char *, char *, char *, char *), op_set set,
					op_dat dat0, int itemSel0, op_map map0, op_access access0,
					op_dat dat1, int itemSel1, op_map map1, op_access access1,
					op_dat dat2, int itemSel2, op_map map2, op_access access2,
					op_dat dat3, int itemSel3, op_map map3, op_access access3,
					op_dat dat4, int itemSel4, op_map map4, op_access access4,
					op_dat dat5, int itemSel5, op_map map5, op_access access5,
					op_dat dat6, int itemSel6, op_map map6, op_access access6,
					op_dat dat7, int itemSel7, op_map map7, op_access access7
					);

void op_par_loop_12 ( void (*subroutineName)(char *, char *, char *, char *, char *, char *, char *, char *, char *, char *, char *, char *), op_set * set,
					  op_dat * dat0, int itemSel0, op_map * map0, op_access access0,
					  op_dat * dat1, int itemSel1, op_map * map1, op_access access1,
					  op_dat * dat2, int itemSel2, op_map * map2, op_access access2,
					  op_dat * dat3, int itemSel3, op_map * map3, op_access access3,
					  op_dat * dat4, int itemSel4, op_map * map4, op_access access4,
					  op_dat * dat5, int itemSel5, op_map * map5, op_access access5,
					  op_dat * dat6, int itemSel6, op_map * map6, op_access access6,
					  op_dat * dat7, int itemSel7, op_map * map7, op_access access7,
					  op_dat * dat8, int itemSel8, op_map * map8, op_access access8,
					  op_dat * dat9, int itemSel9, op_map * map9, op_access access9,
					  op_dat * dat10, int itemSel10, op_map * map10, op_access access10,
					  op_dat * dat11, int itemSel11, op_map * map11, op_access access11
					);


#endif
