void values_operation2(float *values, int *result, int *left, int *right, int *op) {
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
