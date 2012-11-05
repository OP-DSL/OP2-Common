#include "volna_common.h"
#include "op_lib_cpp.h"

void __check_hdf5_error(herr_t err, const char *file, const int line){
  if (err < 0) {
    printf("%s(%i) : OP2_HDF5_error() Runtime API error %d.\n", file,
        line, (int) err);
    exit(-1);
  }
}

int timer_happens(TimerParams *p) {
  int result;
  result = ( p->t <= p->end && p->t >= p->start &&
      p->iter <= p->iend && p->iter >= p->istart );
  result =  result &&
      ( ( p->iter == 0) ||
          ( p->localIter == p->istep ) ||
          ( p->localTime >= p->step ) );
  return result;
}

void read_events_hdf5(hid_t h5file, int num_events, std::vector<TimerParams> *timers, std::vector<EventParams> *events) {
  std::vector<float> timer_start(num_events);
  std::vector<float> timer_end(num_events);
  std::vector<float> timer_step(num_events);
  std::vector<int> timer_istart(num_events);
  std::vector<int> timer_iend(num_events);
  std::vector<int> timer_istep(num_events);

  std::vector<float> event_location_x(num_events);
  std::vector<float> event_location_y(num_events);
  std::vector<int> event_post_update(num_events);
//  std::vector<std::string> event_className(num_events);
//  std::vector<std::string> event_formula(num_events);
//  std::vector<std::string> event_streamName(num_events);

  //const hsize_t num_events_hsize = num_events;
  check_hdf5_error(H5LTread_dataset(h5file, "timer_start", H5T_NATIVE_FLOAT, &timer_start[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_end", H5T_NATIVE_FLOAT, &timer_end[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_step", H5T_NATIVE_FLOAT, &timer_step[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_istart", H5T_NATIVE_INT, &timer_istart[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_iend", H5T_NATIVE_INT, &timer_iend[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "timer_istep", H5T_NATIVE_INT, &timer_istep[0]));

  check_hdf5_error(H5LTread_dataset(h5file, "event_location_x", H5T_NATIVE_FLOAT, &event_location_x[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "event_location_y", H5T_NATIVE_FLOAT, &event_location_y[0]));
  check_hdf5_error(H5LTread_dataset(h5file, "event_post_update", H5T_NATIVE_INT, &event_post_update[0]));

  /*
   * Convert Arrays to AoS
   */
  char buffer[22];
//  char* eventBuffer;
  std::vector<char> eventBuffer;
  int length = 0;
  for (int i = 0; i < num_events; i++) {
    (*timers)[i].start = timer_start[i];
    (*timers)[i].end = timer_end[i];
    (*timers)[i].step = timer_step[i];
    (*timers)[i].istart = timer_istart[i];
    (*timers)[i].iend = timer_iend[i];
    (*timers)[i].istep = timer_istep[i];

    (*events)[i].location_x = event_location_x[i];
    (*events)[i].location_y = event_location_y[i];
    (*events)[i].post_update = event_post_update[i];

    /*
     * If string can not handle a variable size char*, then use the commented lines
     */
    memset(buffer,0,22);
    sprintf(buffer, "event_className%d",i);
    check_hdf5_error(H5LTget_attribute_int(h5file, buffer, "length", &length));
//    eventBuffer = (char*) malloc(length);
    eventBuffer.resize(length);
    check_hdf5_error(H5LTread_dataset_string(h5file, buffer, &eventBuffer[0]));
    (*events)[i].className.assign(&eventBuffer[0], length);
//    free(eventBuffer);

    memset(buffer,0,22);
    sprintf(buffer, "event_formula%d",i);
    check_hdf5_error(H5LTget_attribute_int(h5file, buffer, "length", &length));
//    eventBuffer = (char*)malloc(length);
    eventBuffer.resize(length);
    check_hdf5_error(H5LTread_dataset_string(h5file, buffer, &eventBuffer[0]));
    (*events)[i].formula.assign(&eventBuffer[0], length);
//    free(eventBuffer);

    memset(buffer,0,22);
    sprintf(buffer, "event_streamName%d",i);
    check_hdf5_error(H5LTget_attribute_int(h5file, buffer, "length", &length));
//    eventBuffer = (char*)malloc(length);
    eventBuffer.resize(length);
    check_hdf5_error(H5LTread_dataset_string(h5file, buffer, &eventBuffer[0]));
    (*events)[i].streamName.assign(&eventBuffer[0], length);
//    free(eventBuffer);
  }

}

void processEvents(std::vector<TimerParams> *timers, std::vector<EventParams> *events, int firstTime, int updateTimers, float timeIncrement, int removeFinished, int initPrePost,
    op_set cells, op_dat values, op_dat cellVolumes, op_dat cellCenters, op_dat nodeCoords, op_map cellsToNodes, op_dat temp_initEta, op_dat temp_initBathymetry, BoreParams bore_params, GaussianLandslideParams gaussian_landslide_params) {
  //op_printf("processEvents()... \n");
  int size = (*timers).size();
  int i = 0;
  while (i < size){
    if (timer_happens(&(*timers)[i]) && (initPrePost==2 || (*events)[i].post_update==initPrePost)) {
      if (strcmp((*events)[i].className.c_str(), "InitEta") == 0) {
        InitEta(cells, cellCenters, values, temp_initEta, temp_initEta!=NULL);
      } else if (strcmp((*events)[i].className.c_str(), "InitU") == 0) {
        InitU(cells, cellCenters, values);
      } else if (strcmp((*events)[i].className.c_str(), "InitV") == 0) {
        InitV(cells, cellCenters, values);
      } else if (strcmp((*events)[i].className.c_str(), "InitBathymetry") == 0) {
        InitBathymetry(cells, cellCenters, values, temp_initBathymetry, temp_initBathymetry!=NULL, firstTime);
      } else if (strcmp((*events)[i].className.c_str(), "InitBore") == 0) {
        InitBore(cells, cellCenters, values, bore_params);
      } else if (strcmp((*events)[i].className.c_str(), "InitGaussianLandslide") == 0) {
        InitGaussianLandslide(cells, cellCenters, values, gaussian_landslide_params, firstTime);
      } else if (strcmp((*events)[i].className.c_str(), "OutputTime") == 0) {
        OutputTime(&(*timers)[i]);
        //op_printf("Output iter: %d \n", (*timers)[i].iter);
      } else if (strcmp((*events)[i].className.c_str(), "OutputConservedQuantities") == 0) {
        OutputConservedQuantities(cells, cellVolumes, values);
      } else if (strcmp((*events)[i].className.c_str(), "OutputLocation") == 0) {
        OutputLocation(&(*events)[i], &(*timers)[i], cells, nodeCoords, cellsToNodes, values);
      } else if (strcmp((*events)[i].className.c_str(), "OutputSimulation") == 0) {
        OutputSimulation(1, &(*events)[i], &(*timers)[i], nodeCoords, cellsToNodes, values);
      } else if (strcmp((*events)[i].className.c_str(), "OutputMaxElevation") == 0) {
         OutputMaxElevation(&(*events)[i], &(*timers)[i], nodeCoords, cellsToNodes, values, cells);
      } else {
        op_printf("Unrecognized event %s\n", (*events)[i].className.c_str());
//        exit(-1);
      }
      //timer.LocalReset();
      (*timers)[i].localIter = 0;
      (*timers)[i].localTime = 0;
    }
    if (updateTimers) {
      //op_printf("if(updatesTimers) => true \n");
      //timer.update()
      (*timers)[i].t+= timeIncrement;
      (*timers)[i].iter += 1;
      (*timers)[i].localIter += 1;
      (*timers)[i].localTime += timeIncrement;
    }

    if (removeFinished) {
      //op_printf("if(removeFinished) => true \n");
      //Remove finished events
      if (((*timers)[i].iter >= (*timers)[i].iend) || ((*timers)[i].t >= (*timers)[i].end)) {
        (*timers).erase((*timers).begin()+i);
        (*events).erase((*events).begin()+i);
        size--;
      } else i++;
    } else i++;
  }
}
