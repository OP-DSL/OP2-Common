void SpaceDiscretization_3(double *cell, //OP_RW
              double *CellVolumes //OP_READ
)
{
  cell[0] /= *CellVolumes;
  cell[1] /= *CellVolumes;
  cell[2] /= *CellVolumes;
  cell[3] = 0;
}