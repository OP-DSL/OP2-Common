#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>

#include "op_seq.h"

/* Problem mesh and iterations */
#define FILE_NAME_PATH "new_grid.h5"
#define NUM_ITERATIONS 1000


/* Global Constants */
double gam, gm1, cfl, eps, mach, alpha, qinf[4];

/* wall timer routine */
// Now done using OP2's internal op_timers() call

//outlined elemental kernel - save_soln
inline void save_soln(const double *q, double *qold) {
  for (int n = 0; n < 4; n++)
    qold[n] = q[n];
}

//outlined elemental kernel - adt_calc
inline void adt_calc(const double *x1, const double *x2, const double *x3,
                     const double *x4, const double *q, double *adt) {
  double dx, dy, ri, u, v, c;

  ri = 1.0f / q[0];
  u = ri * q[1];
  v = ri * q[2];
  c = sqrt(gam * gm1 * (ri * q[3] - 0.5f * (u * u + v * v)));

  dx = x2[0] - x1[0];
  dy = x2[1] - x1[1];
  *adt = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x3[0] - x2[0];
  dy = x3[1] - x2[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x4[0] - x3[0];
  dy = x4[1] - x3[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  dx = x1[0] - x4[0];
  dy = x1[1] - x4[1];
  *adt += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

  *adt = (*adt) / cfl;
}

//outlined elemental kernel - res_calc
inline void res_calc(const double *x1, const double *x2, const double *q1,
                     const double *q2, const double *adt1, const double *adt2,
                     double *res1, double *res2) {
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1 * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));
  vol1 = ri * (q1[1] * dy - q1[2] * dx);

  ri = 1.0f / q2[0];
  p2 = gm1 * (q2[3] - 0.5f * ri * (q2[1] * q2[1] + q2[2] * q2[2]));
  vol2 = ri * (q2[1] * dy - q2[2] * dx);

  mu = 0.5f * ((*adt1) + (*adt2)) * eps;

  f = 0.5f * (vol1 * q1[0] + vol2 * q2[0]) + mu * (q1[0] - q2[0]);
  res1[0] += f;
  res2[0] -= f;
  f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * q2[1] + p2 * dy) +
      mu * (q1[1] - q2[1]);
  res1[1] += f;
  res2[1] -= f;
  f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * q2[2] - p2 * dx) +
      mu * (q1[2] - q2[2]);
  res1[2] += f;
  res2[2] -= f;
  f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (q2[3] + p2)) + mu * (q1[3] - q2[3]);
  res1[3] += f;
  res2[3] -= f;
}

//outlined elemental kernel - bres_calc
inline void bres_calc(const double *x1, const double *x2, const double *q1,
                      const double *adt1, double *res1, const int *bound) {
  double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

  dx = x1[0] - x2[0];
  dy = x1[1] - x2[1];

  ri = 1.0f / q1[0];
  p1 = gm1 * (q1[3] - 0.5f * ri * (q1[1] * q1[1] + q1[2] * q1[2]));

  if (*bound == 1) {
    res1[1] += +p1 * dy;
    res1[2] += -p1 * dx;
  } else {
    vol1 = ri * (q1[1] * dy - q1[2] * dx);

    ri = 1.0f / qinf[0];
    p2 = gm1 * (qinf[3] - 0.5f * ri * (qinf[1] * qinf[1] + qinf[2] * qinf[2]));
    vol2 = ri * (qinf[1] * dy - qinf[2] * dx);

    mu = (*adt1) * eps;

    f = 0.5f * (vol1 * q1[0] + vol2 * qinf[0]) + mu * (q1[0] - qinf[0]);
    res1[0] += f;
    f = 0.5f * (vol1 * q1[1] + p1 * dy + vol2 * qinf[1] + p2 * dy) +
        mu * (q1[1] - qinf[1]);
    res1[1] += f;
    f = 0.5f * (vol1 * q1[2] - p1 * dx + vol2 * qinf[2] - p2 * dx) +
        mu * (q1[2] - qinf[2]);
    res1[2] += f;
    f = 0.5f * (vol1 * (q1[3] + p1) + vol2 * (qinf[3] + p2)) +
        mu * (q1[3] - qinf[3]);
    res1[3] += f;
  }
}

//outlined elemental kernel - update
inline void update(const double *qold, double *q, double *res,
                   const double *adt, double *rms) {
  double del, adti;

  adti = 1.0f / (*adt);

  for (int n = 0; n < 4; n++) {
    del = adti * res[n];
    q[n] = qold[n] - del;
    res[n] = 0.0f;
    *rms += del * del;
  }
}


/* main application */
int main(int argc, char **argv) {
  //Initialise the OP2 library, passing runtime args, and setting diagnostics level to low (1)
  op_init(argc, argv, 1);

  int *becell, *ecell, *bound, *bedge, *edge, *cell;
  double *x, *q, *qold, *adt, *res;
  int nnode, ncell, nedge, nbedge, niter;
  double rms;

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  // Load unstructured mesh
  op_printf("***** Load mesh and initialization *****\n");
  char file[] = FILE_NAME_PATH;

  // declare sets
  op_set nodes  = op_decl_set_hdf5(file,  "nodes" );
  op_set edges  = op_decl_set_hdf5(file,  "edges" );
  op_set bedges = op_decl_set_hdf5(file, "bedges");
  op_set cells  = op_decl_set_hdf5(file,  "cells" );

  //declare maps
  op_map pedge   = op_decl_map_hdf5(edges,  nodes, 2, file, "pedge"  );
  op_map pecell  = op_decl_map_hdf5(edges,  cells, 2, file, "pecell" );
  op_map pbedge  = op_decl_map_hdf5(bedges, nodes, 2, file, "pbedge" );
  op_map pbecell = op_decl_map_hdf5(bedges, cells, 1, file, "pbecell");
  op_map pcell   = op_decl_map_hdf5(cells,  nodes, 4, file, "pcell"  );

  //declare data on sets
  op_dat p_bound = op_decl_dat_hdf5(bedges, 1, "int",    file, "p_bound");
  op_dat p_x     = op_decl_dat_hdf5(nodes,  2, "double", file, "p_x"    );
  op_dat p_q     = op_decl_dat_hdf5(cells,  4, "double", file, "p_q"    );
  op_dat p_qold  = op_decl_dat_hdf5(cells,  4, "double", file, "p_qold" );
  op_dat p_adt   = op_decl_dat_hdf5(cells,  1, "double", file, "p_adt"  );
  op_dat p_res   = op_decl_dat_hdf5(cells,  4, "double", file, "p_res"  );

  //read and declare global constants
  op_get_const_hdf5("gam",   1, "double", (char *)&gam,  file);
  op_get_const_hdf5("gm1",   1, "double", (char *)&gm1,  file);
  op_get_const_hdf5("cfl",   1, "double", (char *)&cfl,  file);
  op_get_const_hdf5("eps",   1, "double", (char *)&eps,  file);
  op_get_const_hdf5("alpha", 1, "double", (char *)&alpha,file);
  op_get_const_hdf5("qinf",  4, "double", (char *)&qinf, file);

  op_decl_const(1, "double", &gam  );
  op_decl_const(1, "double", &gm1  );
  op_decl_const(1, "double", &cfl  );
  op_decl_const(1, "double", &eps  );
  op_decl_const(1, "double", &mach );
  op_decl_const(1, "double", &alpha);
  op_decl_const(4, "double", qinf  );

  //get global number of cells 
  ncell = op_get_size(cells);

  //output mesh information
  op_diagnostic_output();

  //partition mesh and create mpi halos
  op_partition("BLOCK", "ANY", edges, pecell, p_x);

  //start timer
  op_timers(&cpu_t1, &wall_t1);

  // main time-marching loop
  op_printf("***** Start Main iteration *************\n");
  for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {

    //save_soln : iterates over cells
    op_par_loop(save_soln, "save_soln", cells,
                op_arg_dat(p_q, -1, OP_ID, 4, "double", OP_READ),
                op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_WRITE));

    // predictor/corrector update loop
    for (int k=0; k < 2; ++k) {

      //adt_calc - calculate area/timstep : iterates over cells
      op_par_loop(adt_calc, "adt_calc", cells,
                  op_arg_dat(p_x,   0, pcell, 2, "double", OP_READ ),
                  op_arg_dat(p_x,   1, pcell, 2, "double", OP_READ ),
                  op_arg_dat(p_x,   2, pcell, 2, "double", OP_READ ),
                  op_arg_dat(p_x,   3, pcell, 2, "double", OP_READ ),
                  op_arg_dat(p_q,  -1, OP_ID, 4, "double", OP_READ ),
                  op_arg_dat(p_adt,-1, OP_ID, 1, "double", OP_WRITE));

      //res_calc - calculate flux residual: iterates over edges
      op_par_loop(res_calc, "res_calc", edges,
                  op_arg_dat(p_x,   0, pedge,  2, "double", OP_READ),
                  op_arg_dat(p_x,   1, pedge,  2, "double", OP_READ),
                  op_arg_dat(p_q,   0, pecell, 4, "double", OP_READ),
                  op_arg_dat(p_q,   1, pecell, 4, "double", OP_READ),
                  op_arg_dat(p_adt, 0, pecell, 1, "double", OP_READ),
                  op_arg_dat(p_adt, 1, pecell, 1, "double", OP_READ),
                  op_arg_dat(p_res, 0, pecell, 4, "double", OP_INC ),
                  op_arg_dat(p_res, 1, pecell, 4, "double", OP_INC ));

      //bres_calc - calculate flux residual in boundary: iterates over boundary edges
      op_par_loop(bres_calc, "bres_calc", bedges,
                  op_arg_dat(p_x,      0, pbedge,  2, "double", OP_READ),
                  op_arg_dat(p_x,      1, pbedge,  2, "double", OP_READ),
                  op_arg_dat(p_q,      0, pbecell, 4, "double", OP_READ),
                  op_arg_dat(p_adt,    0, pbecell, 1, "double", OP_READ),
                  op_arg_dat(p_res,    0, pbecell, 4, "double", OP_INC ),
                  op_arg_dat(p_bound, -1, OP_ID,   1, "int",    OP_READ));

      //update = update flow field - iterates over cells
      rms = 0.0f;
      op_par_loop(update, "update", cells,
                  op_arg_dat(p_qold, -1, OP_ID, 4, "double", OP_READ ),
                  op_arg_dat(p_q,    -1, OP_ID, 4, "double", OP_WRITE),
                  op_arg_dat(p_res,  -1, OP_ID, 4, "double", OP_RW   ),
                  op_arg_dat(p_adt,  -1, OP_ID, 1, "double", OP_READ ),
                  op_arg_gbl(&rms,    1,           "double", OP_INC  ));
    }

    // print iteration history
    rms = sqrt(rms / (double)ncell);
    if (iter % 100 == 0)
      op_printf(" %d  %10.5e \n", iter, rms);
    if (iter % 1000 == 0 && ncell == 720000) {
      float diff = fabs((100.0 * (rms / 0.0001060114637578)) - 100.0);
      op_printf("\nTest problem with %d cells is within %3.15E %% of the "
              "expected solution\n",
              720000, diff);
      if (diff < 0.00001) {
        op_printf("This test is considered PASSED\n");
      } else {
        op_printf("This test is considered FAILED\n");
      }
      op_printf("***** End Main iteration *************\n");
    }
  }

  //end timer
  op_timers(&cpu_t2, &wall_t2);

  // compute and print wall time
  double walltime = wall_t2 - wall_t1;

  op_printf(" Wall time %lf \n", walltime);

  //Finalising the OP2 library
  op_exit();

  return 0;
}
