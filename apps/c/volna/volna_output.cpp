#include "volna_common.h"
#include "getTotalVol.h"

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

//void OutputSimulation(op_set)

//void OutputSimulation::execute( Mesh &mesh, Values &V ) {
//
//  std::stringstream iterStream;
//  iterStream << std::setfill('0') << std::setw(4) << timer.iter;
//  std::string iterString = iterStream.str();
//
//  std::string filename = streamName;
//
//  std::string searchString = "\%i";
//
//  std::string::size_type pos = 0;
//  while ( ( pos = filename.find( searchString, pos ) ) != string::npos ) {
//    filename.replace( pos, searchString.size(), iterString );
//    ++pos;
//  }
//
//  std::ofstream stream( filename.c_str() );
//
//  if ( !stream.good() ) {
//    std::cerr << "Warning: impossible to write in file.";
//  }
//
//  mesh.WriteVTKAscii( stream );
//
//  stream << "CELL_DATA " << mesh.NVolumes << "\n";
//  V.WriteVTKAscii( stream );
//
//  stream.close();
//
//}
