#ifndef op2_mf_init_cg_h
#define op2_mf_init_cg_h

inline void init_cg(double *r, double *c, double *u, double *v, double *p) {
  *c += (*r) * (*r);
  *p = *r;
  *u = 0;
  *v = 0;
}

#endif
