#include "volna_common.h"
#include "getTotalVol.h"

inline void WriteVTKAscii(const char* filename, op_dat nodeCoords, int nnode, op_map cellsToNodes, int ncell, op_dat values) {
  FILE* fp;
  fp = fopen(filename, "w");
  if(fp == NULL) {
    op_printf("can't open file for write %s\n",filename);
    exit(-1);
  }

  // write header
  fprintf(fp,"# vtk DataFile Version 2.0\n Output from OP2 Volna.\n");
  fprintf(fp,"ASCII \nDATASET UNSTRUCTURED_GRID\n\n");

  // write vertices
  fprintf(fp,"POINTS %d float\n", nnode);

  float* nodeCoords_data;
  nodeCoords_data = (float*)nodeCoords->data;

  int i = 0;
//  for (i = 0; i < nodeCoords->size/nodeCoords->dim; ++i) {
  for (i = 0; i < nnode; ++i) {
    fprintf(fp, "%f %f %f \n",
        (float)nodeCoords_data[i*MESH_DIM  ],
        (float)nodeCoords_data[i*MESH_DIM+1],
        0.0f);
  }
  fprintf(fp, "\n");

  // write cells
  fprintf(fp, "CELLS %d %d \n", ncell, 4*ncell);

  for ( i = 0; i < ncell; ++i ) {
    fprintf(fp, "3 %d %d %d \n",
        cellsToNodes->map[i*N_NODESPERCELL  ],
        cellsToNodes->map[i*N_NODESPERCELL+1],
        cellsToNodes->map[i*N_NODESPERCELL+2]);
  }
  fprintf(fp, "\n");

  // write cell types (5 for triangles)
  fprintf(fp, "CELL_TYPES %d \n", ncell);

  for ( i=0; i<ncell; ++i )
    fprintf(fp, "5 \n");
//    os << VTK_TYPE<3,2>::value() << "\n";
  fprintf(fp, "\n");

  float* values_data;
  values_data = (float*) values->data;

  fprintf(fp, "CELL_DATA %d\n"
              "SCALARS Eta double 1\n"
              "LOOKUP_TABLE default\n",
              ncell);
  for ( i=0; i<ncell; ++i )
    fprintf(fp, "%f \n", values_data[i*N_STATEVAR] + values_data[N_STATEVAR+3]);
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS U double 1\n"
              "LOOKUP_TABLE default\n");
  for ( i=0; i<ncell; ++i )
    fprintf(fp, "%f \n", values_data[i*N_STATEVAR+1]);
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS V double 1\n"
              "LOOKUP_TABLE default\n");
  for ( i=0; i<ncell; ++i )
    fprintf(fp, "%f \n", values_data[i*N_STATEVAR+2]);
  fprintf(fp, "\n");

  fprintf(fp, "SCALARS Bathymetry double 1\n"
              "LOOKUP_TABLE default\n");
  for ( i=0; i<ncell; ++i )
    fprintf(fp, "%f \n", values_data[i*N_STATEVAR+3]);
  fprintf(fp, "\n");


  fprintf(fp, "SCALARS Visual double 1\n"
              "LOOKUP_TABLE default\n");
  for ( i=0; i<ncell; ++i ) {
    if(values_data[i*N_STATEVAR] < 1e-3)
      fprintf(fp, "%f \n", 100.0f);
    else
      fprintf(fp, "%f \n", values_data[i*N_STATEVAR] + values_data[N_STATEVAR+3]);
  }
  fprintf(fp, "\n");

  if(fclose(fp) != 0) {
    op_printf("can't close file %s\n",filename);
    exit(-1);
  }
}

void OutputTime(TimerParams *timer) {
  op_printf("Iteration: %d, time: %f \n", (*timer).iter, (*timer).t);
}

void OutputConservedQuantities(op_set cells, op_dat cellVolumes, op_dat values) {
  float totalVol = 0.0f;
  op_par_loop(getTotalVol, "getTotalVol", cells,
      op_arg_dat(cellVolumes, -1, OP_ID, 1, "float", OP_READ),
      op_arg_dat(values, -1, OP_ID, 4, "float", OP_READ),
      op_arg_gbl(&totalVol, 1, "float", OP_INC));

  op_printf("mass(volume): %f \n", totalVol);
}

void OutputSimulation(EventParams *event, TimerParams* timer, op_dat nodeCoords, op_map cellsToNodes, op_dat values) {
  char filename[255];
  strcpy(filename, event->streamName.c_str());
  op_printf("Write output to file: %s \n", filename);
  int nnode = nodeCoords->set->size;
  int ncell = cellsToNodes->from->size;
  const char* substituteIndexPattern = "%i";
  char* pos;
  pos = strstr(filename, substituteIndexPattern);
  char substituteIndex[255];
  sprintf(substituteIndex, "%04d.vtk", timer->iter);
  strcpy(pos, substituteIndex);
  WriteVTKAscii(filename, nodeCoords, nnode, cellsToNodes, ncell, values);
}

double normcomp(op_dat dat, int off) {
  int dim = dat->dim;
  float *data = (float *)(dat->data);
  double norm = 0.0;
  for (int i = 0; i < dat->set->size; i++) {
    norm += data[dim*i + off]*data[dim*i + off];
  }
  return sqrt(norm);
}
