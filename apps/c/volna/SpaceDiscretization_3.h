void SpaceDiscretization_3(double *H, double *U, double *V, //OP_RW
              double *Zb, //OP_WRITE
              double *CellVolumes //OP_READ
)
{
  *H /= *CellVolumes;
  *U /= *CellVolumes;
  *V /= *CellVolumes;
  *Zb = 0;
}