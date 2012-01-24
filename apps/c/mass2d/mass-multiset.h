void mass_multiset(float *A, float *x, int i, int j, int q)
{
    float J[2][2];
    float detJ;
    const float w[3] = {0.166667, 0.166667, 0.166667};
    const float CG1[3][3] = {{0.666667, 0.166667, 0.166667},
                             {0.166667, 0.666667, 0.166667},
                             {0.166667, 0.166667, 0.666667}};

    J[0][0] = x[2] - x[0];
    J[0][1] = x[4] - x[0];
    J[0][1] = x[3] - x[1];
    J[1][1] = x[5] - x[1];

    detJ = J[0][0] * J[1][1] - J[0][1] * J[1][0];

    *A += CG1[i][q] * CG1[j][q] * detJ * w[q];
}
