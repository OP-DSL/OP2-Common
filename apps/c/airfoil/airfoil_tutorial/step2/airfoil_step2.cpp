#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <sys/time.h>

#include "op_seq.h"

/* Problem mesh and iterations */
#define FILE_NAME_PATH "new_grid.dat"
#define NUM_ITERATIONS 1000


/* Global Constants */
#define CONST_GAM 1.4f
#define CONST_MACH 0.4f
#define CONST_P 1.0f
#define CONST_R 1.0f
const double u = sqrt(CONST_GAM * CONST_P / CONST_R) * CONST_MACH;
const double e = CONST_P / (CONST_R * (CONST_GAM - 1.0f)) + 0.5f * u * u;
const double gam = CONST_GAM;
const double gm1 = CONST_GAM - 1.0f;
const double cfl = 0.9f;
const double eps = 0.05f;
const double alpha = 3.0f * atan(1.0f) / 45.0f;
const double qinf[4] = {CONST_R, (CONST_R * u), 0.0f, (CONST_R * e)};

/* wall timer routine */
void timer(double *cpu, double *et) {
  (void)cpu;
  struct timeval t;

  gettimeofday(&t, (struct timezone *)0);
  *et = t.tv_sec + t.tv_usec * 1.0e-6;
}

/* main application */
int main(int argc, char **argv) {
  //Initialise the OP2 library, passing runtime args, and setting diagnostics level to low (1)
  op_init(argc, argv, 2);

  int *becell, *ecell, *bound, *bedge, *edge, *cell;
  double *x, *q, *qold, *adt, *res;
  int nnode, ncell, nedge, nbedge, niter;
  double rms;

  // timer
  double cpu_t1, cpu_t2, wall_t1, wall_t2;

  // Load unstructured mesh
  printf("***** Load mesh and initialization *****\n");
  FILE *fp;
  if ((fp = fopen(FILE_NAME_PATH, "r")) == NULL) {
    printf("can't open file FILE_NAME_PATH\n");
    exit(-1);
  }
  if (fscanf(fp, "%d %d %d %d \n", &nnode, &ncell, &nedge, &nbedge) != 4) {
    printf("error reading from FILE_NAME_PATH\n");
    exit(-1);
  }

  cell   = (int *)malloc(4 * ncell  * sizeof(int));
  edge   = (int *)malloc(2 * nedge  * sizeof(int));
  ecell  = (int *)malloc(2 * nedge  * sizeof(int));
  bedge  = (int *)malloc(2 * nbedge * sizeof(int));
  becell = (int *)malloc(1 * nbedge * sizeof(int));
  bound  = (int *)malloc(1 * nbedge * sizeof(int));
  x      = (double *)malloc(2 * nnode * sizeof(double));
  q      = (double *)malloc(4 * ncell * sizeof(double));
  qold   = (double *)malloc(4 * ncell * sizeof(double));
  res    = (double *)malloc(4 * ncell * sizeof(double));
  adt    = (double *)malloc(1 * ncell * sizeof(double));

  for (int n = 0; n < nnode; n++) {
    if (fscanf(fp, "%lf %lf \n", &x[2 * n], &x[2 * n + 1]) != 2) {
      printf("error reading from FILE_NAME_PATH\n");
      exit(-1);
    }
  }
  for (int n = 0; n < ncell; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &cell[4 * n], &cell[4 * n + 1],
               &cell[4 * n + 2], &cell[4 * n + 3]) != 4) {
      printf("error reading from FILE_NAME_PATH\n");
      exit(-1);
    }
  }
  for (int n = 0; n < nedge; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &edge[2 * n], &edge[2 * n + 1],
               &ecell[2 * n], &ecell[2 * n + 1]) != 4) {
      printf("error reading from FILE_NAME_PATH\n");
      exit(-1);
    }
  }
  for (int n = 0; n < nbedge; n++) {
    if (fscanf(fp, "%d %d %d %d \n", &bedge[2 * n], &bedge[2 * n + 1],
               &becell[n], &bound[n]) != 4) {
      printf("error reading from FILE_NAME_PATH\n");
      exit(-1);
    }
  }
  fclose(fp);

  for (int n = 0; n < ncell; n++) {
    for (int m = 0; m < 4; m++) {
      q[4 * n + m] = qinf[m];
      res[4 * n + m] = 0.0f;
    }
  }

  // declare sets
  op_set nodes  = op_decl_set(nnode,  "nodes" );
  op_set edges  = op_decl_set(nedge,  "edges" );
  op_set bedges = op_decl_set(nbedge, "bedges");
  op_set cells  = op_decl_set(ncell,  "cells" );

  //declare maps
  op_map pedge   = op_decl_map(edges,  nodes, 2, edge,   "pedge"  );
  op_map pecell  = op_decl_map(edges,  cells, 2, ecell,  "pecell" );
  op_map pbedge  = op_decl_map(bedges, nodes, 2, bedge,  "pbedge" );
  op_map pbecell = op_decl_map(bedges, cells, 1, becell, "pbecell");
  op_map pcell   = op_decl_map(cells,  nodes, 4, cell,   "pcell"  );

  //declare data on sets
  op_dat p_bound = op_decl_dat(bedges, 1, "int",    bound, "p_bound");
  op_dat p_x     = op_decl_dat(nodes,  2, "double", x,     "p_x"    );
  op_dat p_q     = op_decl_dat(cells,  4, "double", q,     "p_q"    );
  op_dat p_qold  = op_decl_dat(cells,  4, "double", qold,  "p_qold" );
  op_dat p_adt   = op_decl_dat(cells,  1, "double", adt,   "p_adt"  );
  op_dat p_res   = op_decl_dat(cells,  4, "double", res,   "p_res"  );

  //declare global constants
  op_decl_const(1, "double", &gam  );
  op_decl_const(1, "double", &gm1  );
  op_decl_const(1, "double", &cfl  );
  op_decl_const(1, "double", &eps  );
  op_decl_const(1, "double", &alpha);
  op_decl_const(4, "double", qinf  );

  //output mesh information
  op_diagnostic_output();

  //start timer
  timer(&cpu_t1, &wall_t1);

  // main time-marching loop
  printf("***** Start Main iteration *************\n");
  for (int iter = 1; iter <= NUM_ITERATIONS; iter++) {

    //save_soln : iterates over cells
    for (int iteration = 0; iteration < (ncell * 4); ++iteration) {
      qold[iteration] = q[iteration];
    }

    // predictor/corrector update loop
    for (int k=0; k < 2; ++k) {

      //adt_calc - calculate area/timstep : iterates over cells
      for (int iteration = 0; iteration < ncell; ++iteration) {
        int map1idx = cell[iteration * 4 + 0];
        int map2idx = cell[iteration * 4 + 1];
        int map3idx = cell[iteration * 4 + 2];
        int map4idx = cell[iteration * 4 + 3];

        double dx, dy, ri, u, v, c;

        ri = 1.0f / q[4 * iteration + 0];
        u = ri * q[4 * iteration + 1];
        v = ri * q[4 * iteration + 2];
        c = sqrt(gam * gm1 * (ri * q[4 * iteration + 3] - 0.5f * (u * u + v * v)));

        dx = x[2 * map2idx + 0] - x[2 * map1idx + 0];
        dy = x[2 * map2idx + 1] - x[2 * map1idx + 1];
        adt[iteration] = fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

        dx = x[2 * map3idx + 0] - x[2 * map2idx + 0];
        dy = x[2 * map3idx + 1] - x[2 * map2idx + 1];
        adt[iteration] += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

        dx = x[2 * map4idx + 0] - x[2 * map3idx + 0];
        dy = x[2 * map4idx + 1] - x[2 * map3idx + 1];
        adt[iteration] += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

        dx = x[2 * map1idx + 0] - x[2 * map4idx + 0];
        dy = x[2 * map1idx + 1] - x[2 * map4idx + 1];
        adt[iteration] += fabs(u * dy - v * dx) + c * sqrt(dx * dx + dy * dy);

        adt[iteration] = (adt[iteration]) / cfl;
      }


      //res_calc - calculate flux residual: iterates over edges
      for (int iteration = 0; iteration < nedge; ++iteration) {
        int map1idx = edge[iteration * 2 + 0];
        int map2idx = edge[iteration * 2 + 1];
        int map3idx = ecell[iteration * 2 + 0];
        int map4idx = ecell[iteration * 2 + 1];

        double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

        dx = x[2 * map1idx + 0] - x[2 * map2idx + 0];
        dy = x[2 * map1idx + 1] - x[2 * map2idx + 1];

        ri = 1.0f / q[4 * map3idx + 0];
        p1 = gm1 * (q[4 * map3idx + 3] - 0.5f * ri * (q[4 * map3idx + 1] * q[4 * map3idx + 1] + q[4 * map3idx + 2] * q[4 * map3idx + 2]));
        vol1 = ri * (q[4 * map3idx + 1] * dy - q[4 * map3idx + 2] * dx);

        ri = 1.0f / q[4 * map4idx + 0];
        p2 = gm1 * (q[4 * map4idx + 3] - 0.5f * ri * (q[4 * map4idx + 1] * q[4 * map4idx + 1] + q[4 * map4idx + 2] * q[4 * map4idx + 2]));
        vol2 = ri * (q[4 * map4idx + 1] * dy - q[4 * map4idx + 2] * dx);

        mu = 0.5f * ((adt[1 * map3idx]) + (adt[1 * map4idx])) * eps;

        f = 0.5f * (vol1 * q[4 * map3idx + 0] + vol2 * q[4 * map4idx + 0]) + mu * (q[4 * map3idx + 0] - q[4 * map4idx + 0]);
        res[4 * map3idx + 0] += f;
        res[4 * map4idx + 0] -= f;
        f = 0.5f * (vol1 * q[4 * map3idx + 1] + p1 * dy + vol2 * q[4 * map4idx + 1] + p2 * dy) +
            mu * (q[4 * map3idx + 1] - q[4 * map4idx + 1]);
        res[4 * map3idx + 1] += f;
        res[4 * map4idx + 1] -= f;
        f = 0.5f * (vol1 * q[4 * map3idx + 2] - p1 * dx + vol2 * q[4 * map4idx + 2] - p2 * dx) +
            mu * (q[4 * map3idx + 2] - q[4 * map4idx + 2]);
        res[4 * map3idx + 2] += f;
        res[4 * map4idx + 2] -= f;
        f = 0.5f * (vol1 * (q[4 * map3idx + 3] + p1) + vol2 * (q[4 * map4idx + 3] + p2)) + mu * (q[4 * map3idx + 3] - q[4 * map4idx + 3]);
        res[4 * map3idx + 3] += f;
        res[4 * map4idx + 3] -= f;
      }

      //bres_calc - calculate flux residual in boundary: iterates over boundary edges
      for (int iteration = 0; iteration < nbedge; ++iteration) {
        int map1idx = bedge[iteration * 2 + 0];
        int map2idx = bedge[iteration * 2 + 1];
        int map3idx = becell[iteration * 1 + 0];

        double dx, dy, mu, ri, p1, vol1, p2, vol2, f;

        dx = x[2 * map1idx + 0] - x[2 * map2idx + 0];
        dy = x[2 * map1idx + 1] - x[2 * map2idx + 1];

        ri = 1.0f / q[4 * map3idx + 0];
        p1 = gm1 * (q[4 * map3idx + 3] - 0.5f * ri * (q[4 * map3idx + 1] * q[4 * map3idx + 1] + q[4 * map3idx + 2] * q[4 * map3idx + 2]));

        if (bound[iteration] == 1) {
          res[4 * map3idx + 1] += +p1 * dy;
          res[4 * map3idx + 2] += -p1 * dx;
        } else {
          vol1 = ri * (q[4 * map3idx + 1] * dy - q[4 * map3idx + 2] * dx);

          ri = 1.0f / qinf[0];
          p2 = gm1 * (qinf[3] - 0.5f * ri * (qinf[1] * qinf[1] + qinf[2] * qinf[2]));
          vol2 = ri * (qinf[1] * dy - qinf[2] * dx);

          mu = (adt[1 * map3idx]) * eps;

          f = 0.5f * (vol1 * q[4 * map3idx + 0] + vol2 * qinf[0]) + mu * (q[4 * map3idx + 0] - qinf[0]);
          res[4 * map3idx + 0] += f;
          f = 0.5f * (vol1 * q[4 * map3idx + 1] + p1 * dy + vol2 * qinf[1] + p2 * dy) +
              mu * (q[4 * map3idx + 1] - qinf[1]);
          res[4 * map3idx + 1] += f;
          f = 0.5f * (vol1 * q[4 * map3idx + 2] - p1 * dx + vol2 * qinf[2] - p2 * dx) +
              mu * (q[4 * map3idx + 2] - qinf[2]);
          res[4 * map3idx + 2] += f;
          f = 0.5f * (vol1 * (q[4 * map3idx + 3] + p1) + vol2 * (qinf[3] + p2)) +
              mu * (q[4 * map3idx + 3] - qinf[3]);
          res[4 * map3idx + 3] += f;
        }
      }

      //update = update flow field - iterates over cells
      rms = 0.0f;
      for (int iteration = 0; iteration < ncell; ++iteration) {
        double del, adti;

        adti = 1.0f / (adt[iteration]);

        for (int n = 0; n < 4; n++) {
          del = adti * res[iteration * 4 + n];
          q[iteration * 4 + n] = qold[iteration * 4 + n] - del;
          res[iteration * 4 + n] = 0.0f;
          rms += del * del;
        }
      }
    }

    // print iteration history
    rms = sqrt(rms / (double)ncell);
    if (iter % 100 == 0)
      printf(" %d  %10.5e \n", iter, rms);
    if (iter % 1000 == 0 && ncell == 720000) {
      float diff = fabs((100.0 * (rms / 0.0001060114637578)) - 100.0);
      printf("\nTest problem with %d cells is within %3.15E %% of the "
              "expected solution\n",
              720000, diff);
      if (diff < 0.00001) {
        printf("This test is considered PASSED\n");
      } else {
        printf("This test is considered FAILED\n");
      }
      printf("***** End Main iteration *************\n");
    }
  }

  //end timer
  timer(&cpu_t2, &wall_t2);

  // compute and print wall time
  double walltime = wall_t2 - wall_t1;
  printf(" Wall time %lf \n", walltime);

  free(cell);
  free(edge);
  free(ecell);
  free(bedge);
  free(becell);
  free(bound);
  free(x);
  free(q);
  free(qold);
  free(adt);
  free(res);

  //Finalising the OP2 library
  op_exit();

  return 0;
}
