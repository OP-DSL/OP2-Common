#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "op_seq.h"

//
// Define meta data
//
#define N_STATEVAR 4
#define MESH_DIM 2
#define N_NODESPERCELL 3
#define N_CELLSPEREDGE 2

struct RectangleDomainParams {
  int nx, ny;
  double xmin, xmax, ymin, ymax;
};

struct BoreParams {
  double x0, Hl, ul, vl;
  double S;
};

//these are not const, we just don't want to pass them around
extern double timestamp;
extern int itercount;

//constants
extern double EPS, CFL, g;
extern op_dat currentMaxElevation;

struct GaussianLandslideParams {
  double A, v, lx, ly, mesh_xmin;//TODO: mesh_xmin compute
};

struct TimerParams {
  double start, end, step, localTime, t;
  unsigned int istart, iend, istep, localIter, iter;
};

struct EventParams {
  double location_x, location_y;
  int post_update;
  std::string className;
  std::string formula;
  std::string streamName;
};

int timer_happens(TimerParams *p);
void read_events_hdf5(hid_t h5file, int num_events, std::vector<TimerParams> *timers, std::vector<EventParams> *events);
void processEvents(std::vector<TimerParams> *timers, std::vector<EventParams> *events, int firstTime, int updateTimers, double timeIncrement, int removeFinished, int initPrePost,
                   op_set cells, op_dat values, op_dat cellVolumes, op_dat cellCenters, op_dat nodeCoords, op_map cellsToNodes, op_dat temp_initEta, op_dat temp_initBathymetry, BoreParams bore_params, GaussianLandslideParams gaussian_landslide_params);

void InitEta(op_set cells, op_dat cellCenters, op_dat values, op_dat temp_initEta, int fromFile);
void InitU(op_set cells, op_dat cellCenters, op_dat values);
void InitV(op_set cells, op_dat cellCenters, op_dat values);
void InitBathymetry(op_set cells, op_dat cellCenters, op_dat values, op_dat temp_initBathymetry, int fromFile, int firstTime);
void InitBore(op_set cells, op_dat cellCenters, op_dat values, BoreParams params);
void InitGaussianLandslide(op_set cells, op_dat cellCenters, op_dat values, GaussianLandslideParams params, int firstTime);

void OutputTime(TimerParams *timer);
void OutputConservedQuantities(op_set cells, op_dat cellVolumes, op_dat values);
void OutputSimulation( EventParams *event, TimerParams* timer, op_dat nodeCoords, op_map cellsToNodes, op_dat values);
void OutputMaxElevation(EventParams *event, TimerParams* timer, op_dat nodeCoords, op_map cellsToNodes, op_dat values, op_set cells);
double normcomp(op_dat dat, int off);
void dumpme(op_dat dat, int off);

void spaceDiscretization(op_dat data_in, op_dat data_out, double *minTimestep,
    op_dat leftCellValues, op_dat rightCellValues, op_dat interfaceBathy,
    op_dat bathySource, op_dat edgeFluxes, op_dat maxEdgeEigenvalues,
    op_dat edgeNormals, op_dat edgeLength, op_dat cellVolumes, op_dat isBoundary,
    op_set cells, op_set edges, op_map edgesToCells, op_map cellsToEdges, int most);

//
//helper functions
//
#define check_hdf5_error(err) __check_hdf5_error(err, __FILE__, __LINE__)
void __check_hdf5_error(herr_t err, const char *file, const int line);
