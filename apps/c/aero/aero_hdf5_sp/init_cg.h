#ifndef op2_mf_init_cg_h
#define op2_mf_init_cg_h

inline void init_cg(const float *r, float *c, float *u, float *v, float *p) {
  *c += (*r) * (*r);
  *p = *r;
  *u = 0.0f;
  *v = 0.0f;
}

#endif
