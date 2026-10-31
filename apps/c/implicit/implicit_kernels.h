#pragma once

// Backward Euler for u_t = div(K grad u) - C u with NV components, scaled by
// h^2: diagonal blocks h^2/dt I + h^2 C + 4 K, off-diagonal blocks -K.

inline void diag_block(const int *nbnd, double *A) {
    for (int r = 0; r < NV; r++)
        for (int c = 0; c < NV; c++)
            A[r * NV + c] += (r == c ? h2dt : 0.0) + h2 * Cc[r * NV + c] + nbnd[0] * Kc[r * NV + c];
}

inline void edge_blocks(double *A00, double *A01, double *A10, double *A11) {
    for (int k = 0; k < NV * NV; k++) {
        A00[k] += Kc[k];
        A01[k] -= Kc[k];
        A10[k] -= Kc[k];
        A11[k] += Kc[k];
    }
}

inline void rhs(const double *u, double *b) {
    for (int v = 0; v < NV; v++)
        b[v] = h2dt * u[v];
}

inline void initial(const double *xy, const double *w, double *u) {
    double s = sin(3.14159265358979323846 * xy[0]) * sin(3.14159265358979323846 * xy[1]);
    for (int v = 0; v < NV; v++)
        u[v] = w[v] * s;
}

inline void bump(const double *xy, const double *w, double *u) {
    for (int v = 0; v < NV; v++) {
        double dx = xy[0] - 0.3 - 0.1 * v, dy = xy[1] - 0.6;
        u[v] = w[v] * exp(-40.0 * (dx * dx + dy * dy));
    }
}

inline void residual_norms(const double *ax, const double *b, double *r2, double *b2) {
    for (int v = 0; v < NV; v++) {
        *r2 += (ax[v] - b[v]) * (ax[v] - b[v]);
        *b2 += b[v] * b[v];
    }
}

inline void error_norms(const double *u, const double *xy, const double *g, double *e2, double *emax) {
    double s = sin(3.14159265358979323846 * xy[0]) * sin(3.14159265358979323846 * xy[1]);
    for (int v = 0; v < NV; v++) {
        double e = u[v] - g[v] * s;
        *e2 += e * e;
        *emax = fmax(*emax, fabs(e));
    }
}

// Block Jacobi, from the diagonal blocks in solver memory: Gauss-Jordan,
// which needs no pivoting for these diagonally dominant blocks.
inline void invert_block(const double *A, double *Dinv) {
    double a[NV * NV];
    for (int k = 0; k < NV * NV; k++) {
        a[k] = A[k];
        Dinv[k] = 0.0;
    }
    for (int k = 0; k < NV; k++)
        Dinv[k * NV + k] = 1.0;
    for (int k = 0; k < NV; k++) {
        double p = 1.0 / a[k * NV + k];
        for (int c = 0; c < NV; c++) {
            a[k * NV + c] *= p;
            Dinv[k * NV + c] *= p;
        }
        for (int r = 0; r < NV; r++) {
            if (r == k)
                continue;
            double f = a[r * NV + k];
            for (int c = 0; c < NV; c++) {
                a[r * NV + c] -= f * a[k * NV + c];
                Dinv[r * NV + c] -= f * Dinv[k * NV + c];
            }
        }
    }
}

inline void apply_block(const double *Dinv, const double *r, double *z) {
    for (int i = 0; i < NV; i++) {
        double s = 0.0;
        for (int j = 0; j < NV; j++)
            s += Dinv[i * NV + j] * r[j];
        z[i] = s;
    }
}

// y = A x without the matrix: the node part, then each edge's.
inline void apply_node(const int *nbnd, const double *x, double *y) {
    for (int r = 0; r < NV; r++) {
        double s = h2dt * x[r];
        for (int c = 0; c < NV; c++)
            s += (h2 * Cc[r * NV + c] + nbnd[0] * Kc[r * NV + c]) * x[c];
        y[r] = s;
    }
}

inline void apply_edge(const double *x0, const double *x1, double *y0, double *y1) {
    for (int r = 0; r < NV; r++) {
        double s = 0.0;
        for (int c = 0; c < NV; c++)
            s += Kc[r * NV + c] * (x0[c] - x1[c]);
        y0[r] += s;
        y1[r] -= s;
    }
}
