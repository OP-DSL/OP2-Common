// From the MCFC identity test case - the only changes are that the dt parameter
// has been removed, so the argument lists match, and "double" has been changed
// to "float".
void mass(float* localTensor, float* c0[2], int i_r_0, int i_r_1)
{
  const float CG1[3][6] = { {  0.09157621, 0.09157621, 0.81684757,
                               0.44594849, 0.44594849, 0.10810302 },
                             {  0.09157621, 0.81684757, 0.09157621,
                               0.44594849, 0.10810302, 0.44594849 },
                             {  0.81684757, 0.09157621, 0.09157621,
                               0.10810302, 0.44594849, 0.44594849 } };
  const float d_CG1[3][6][2] = { { {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. },
                                   {  1., 0. } },

                                  { {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. },
                                   {  0., 1. } },

                                  { { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. },
                                   { -1.,-1. } } };
  const float w[6] = {  0.05497587, 0.05497587, 0.05497587, 0.11169079,
                         0.11169079, 0.11169079 };
  float c_q0[3][2][2];
  for(int i_g = 0; i_g < 3; i_g++)
  {
    for(int i_d_0 = 0; i_d_0 < 2; i_d_0++)
    {
      for(int i_d_1 = 0; i_d_1 < 2; i_d_1++)
      {
        c_q0[i_g][i_d_0][i_d_1] = 0.0;
        for(int q_r_0 = 0; q_r_0 < 3; q_r_0++)
        {
          c_q0[i_g][i_d_0][i_d_1] += c0[q_r_0][i_d_0] * d_CG1[q_r_0][i_g][i_d_1];
        };
      };
    };
  };
  for(int i_g = 0; i_g < 3; i_g++)
  {
    float ST0 = 0.0;
    ST0 += CG1[i_r_0][i_g] * CG1[i_r_1][i_g] * (c_q0[i_g][0][0] * c_q0[i_g][1][1] + -1 * c_q0[i_g][0][1] * c_q0[i_g][1][0]);
    localTensor[0] += ST0 * w[i_g];
  };
}
