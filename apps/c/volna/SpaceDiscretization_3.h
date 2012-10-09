void SpaceDiscretization_3(float *cell, //OP_RW
              float *CellVolumes //OP_READ
)
{
  cell[0] /= *CellVolumes;
  cell[1] /= *CellVolumes;
  cell[2] /= *CellVolumes;
  cell[3] = 0;
}