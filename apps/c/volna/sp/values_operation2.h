inline void values_operation2(float *values, const int *result, const int *left, const int *right, const int *op) {
  switch (*op) {
  case 0:
    values[*result] = values[*left] + values[*right];
    break;
  case 1:
    values[*result] = values[*left] - values[*right];
    break;
  case 2:
    values[*result] = values[*left] * values[*right];
    break;
  case 3:
    values[*result] = values[*left] / values[*right];
    break;
  }
}
