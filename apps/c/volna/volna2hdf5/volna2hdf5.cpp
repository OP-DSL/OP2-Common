//============================================================================
// Name        : volna2hdf5.cpp
// Author      : Endre Laszlo
// Version     :
// Copyright   :
// Description :
//============================================================================

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include <fstream>
#include <iostream>
#include <string>
#include <map>

//
// Sequential OP2 function declarations
//
#include "op_seq.h"

//
// VOLNA function declarations
//
#include "simulation.hpp"
#include "paramFileParser.hpp"

//
// Define meta data
//
#define N_STATEVAR 3
#define MESH_DIM 2
#define N_NODESPERCELL 3
#define N_CELLSPEREDGE 2

//
//helper functions
//
#define check_hdf5_error(err)           __check_hdf5_error      (err, __FILE__, __LINE__)
void __check_hdf5_error(herr_t err, const char *file, const int line) {
  if (err < 0) {
    printf("%s(%i) : OP2_HDF5_error() Runtime API error %d.\n", file, line, (int)err);
      exit(-1);
  }
}


int main(int argc, char **argv) {
	if(argc != 2) {
		printf("Wrong parameters! Please specify the VOLNA configuration script filename with the *.vln extension, e.g. ./volna2hdf5 bump.vln \n");
		exit(-1);
	}

	//
	////////////// INIT VOLNA TO GAIN DATA IMPORT //////////////
	//

	printf("Importing data from VOLNA Framework ...\n");
	//op_mesh_io_import(filename_msh, 2, 3, &x, &cell, &edge, &ecell, &bedge, &becell, &bound, &nnode, &ncell, &nedge, &nbedge);

	char const * const file = argv[1];
	spirit::file_iterator<> file_it( file );

	Simulation sim;
	ParamFileController fileParser;

	std::cerr << file << "\n";
	std::cerr << 	"EPSILON value: " << EPS << " \n";
	std::cerr << "Parsing parameter file... ";

	fileParser.parse( file_it, file_it.make_end(), sim );
	//sim.MeshFileName = "stlaurent_35k.msh";

	std::cerr << "Final time is : " << sim.FinalTime << std::endl;
	std::cerr << "Mesh fileName is : " << sim.MeshFileName << std::endl;
	std::cerr << "Initial value formula is : '" << sim.InitFormulas.eta
			<< "'" << std::endl;
	std::cerr << "Bathymetry formula is : '" << sim.InitFormulas.bathymetry
			<< "'" << std::endl;
	std::cerr << "Horizontal velocity formula is : '" << sim.InitFormulas.U
			<< "'" << std::endl;
	std::cerr << "Vertical velocity formula is : '" << sim.InitFormulas.V
			<< "'" << std::endl;

	// Initialize simulation: load mesh, calculate geometry data
	sim.init();

	//
	////////////// INITIALIZE OP2 DATA /////////////////////
	//

	op_printf("Initializing OP2...\n");
	op_init(argc, argv, 2);

	int *cell = NULL; // Node IDs of cells
	int *ecell = NULL; // Cell IDs of edge
	int *ccell = NULL; // Cell IDs of neighbouring cells
	int *cedge = NULL; // Edge IDs of cells
	double *ccent = NULL; // Cell centre vectors
	double *carea = NULL; // Cell area
	double *enorm = NULL; // Edge normal vectors. Pointing from left cell to the right???
	double *ecent = NULL; // Edge center vectors
	double *eleng = NULL; // Edge length
	double *x = NULL; // Node coordinates in 2D
	//double *w = NULL; // Conservative variables
	//double *wold = NULL; // Old conservative variables
	//double *res = NULL; // Per edge residuals
	//double *dt = NULL; // Time step

	int nnode=0, ncell=0, nedge=0, niter=0; // Number of nodes, cells, edges and iterations

	nnode = sim.mesh.NPoints; // Use this variable to obtain the no. nodes. E.g. sim.mesh.Nodes.size() would return invalid data
	ncell = sim.mesh.NVolumes;
	nedge = sim.mesh.NFaces;

	printf("GMSH file data statistics: \n");
	printf("  No. nodes = %d\n",nnode);
	printf("  No. cells = %d\n",ncell);
	printf("Connectivity data statistics: \n");
	printf("  No. of edges           = %d\n", nedge);
	//printf("  No. of boundary edges  = %d\n\n", nbedge);


	// Arrays for mapping data
	cell = (int*) malloc(N_NODESPERCELL * ncell * sizeof(int));
	ecell = (int*) malloc(N_CELLSPEREDGE * nedge * sizeof(int));
	ccell = (int*) malloc(N_NODESPERCELL * ncell * sizeof(int));
	cedge = (int*) malloc(N_NODESPERCELL * ncell * sizeof(int));
	ccent = (double*) malloc(MESH_DIM * ncell * sizeof(double));
	carea = (double*) malloc(ncell * sizeof(double));
	enorm = (double*) malloc(MESH_DIM * nedge * sizeof(double));
	ecent = (double*) malloc(MESH_DIM * nedge * sizeof(double));
	eleng = (double*) malloc(nedge * sizeof(double));
	x = (double*) malloc(MESH_DIM * nnode * sizeof(double));
	//w = (double*) malloc(N_STATEVAR * ncell * sizeof(double));
	//wold = (double*) malloc(N_STATEVAR * ncell * sizeof(double));
	//res = (double*) malloc(N_STATEVAR * nedge * sizeof(double));



	//
	////////////// USE VOLNA FOR DATA IMPORT //////////////
	//

	int i = 0;
	std::cout << "Number of nodes according to sim.mesh.Nodes.size() = " << sim.mesh.Nodes.size() << std::endl;
	std::cout << "Number of nodes according to sim.mesh.NPoints      = " << sim.mesh.NPoints << "  <=== Accept this one! " << std::endl;

	// Import node coordinates
	for(i=0; i<sim.mesh.NPoints; i++) {
		x[i*MESH_DIM  ] = sim.mesh.Nodes[i].x();
		x[i*MESH_DIM+1] = sim.mesh.Nodes[i].y();
		//std::cout << i << "  x,y,z = " << sim.mesh.Nodes[i].x() << " " << sim.mesh.Nodes[i].y() << " " << sim.mesh.Nodes[i].z() << endl;
	}

	// Boost arrays for temporarly storing mesh data
	boost::array<int,N_NODESPERCELL> vertices;
	boost::array<int,N_NODESPERCELL> neighbors;
	boost::array<int,N_NODESPERCELL> facet_ids;
	for(i=0; i<sim.mesh.NVolumes; i++) {

		vertices  = sim.mesh.Cells[i].vertices();
		neighbors = sim.mesh.Cells[i].neighbors();
		facet_ids = sim.mesh.Cells[i].facets();

		cell[i*N_NODESPERCELL  ] = vertices[0];
		cell[i*N_NODESPERCELL+1] = vertices[1];
		cell[i*N_NODESPERCELL+2] = vertices[2];

		ccell[i*N_NODESPERCELL  ] = neighbors[0];
		ccell[i*N_NODESPERCELL+1] = neighbors[1];
		ccell[i*N_NODESPERCELL+2] = neighbors[2];

		cedge[i*N_NODESPERCELL  ] = facet_ids[0];
		cedge[i*N_NODESPERCELL+1] = facet_ids[1];
		cedge[i*N_NODESPERCELL+2] = facet_ids[2];

		ccent[i*N_NODESPERCELL  ] = sim.mesh.CellCenters.x(i);
		ccent[i*N_NODESPERCELL+1] = sim.mesh.CellCenters.y(i);

		carea[i] = sim.mesh.CellVolumes(i);



		//    	std::cout << "Cell " << i << " nodes = " << vertices[0] << " " << vertices[1] << " " << vertices[2] << std::endl;
		//    	std::cout << "Cell " << i << " neighbours = " << neighbors[0] << " "<< neighbors[1] << " " << neighbors[2] <<  std::endl;
		//    	std::cout << "Cell " << i << " facets  = " << facet_ids[0] << " "<< facet_ids[1] << " " << facet_ids[2] <<  std::endl;
		//    	std::cout << "Cell " << i << " center  = [ " << ccent[i*N_NODESPERCELL] << " , "<< ccent[i*N_NODESPERCELL+1] << " ]" <<  std::endl;
		//    	std::cout << "Cell " << i << " area  = " << carea[i] <<  std::endl;
	}

	// Store edge data: edge-cell map, edge normal vectors
	for(i=0; i<sim.mesh.NFaces; i++) {
		ecell[i*N_CELLSPEREDGE  ] = sim.mesh.Facets[i].LeftCell();
		ecell[i*N_CELLSPEREDGE+1] = sim.mesh.Facets[i].RightCell();

		enorm[i*N_CELLSPEREDGE  ] = sim.mesh.FacetNormals.x(i);
		enorm[i*N_CELLSPEREDGE+1] = sim.mesh.FacetNormals.y(i);

		ecent[i*N_CELLSPEREDGE  ] = sim.mesh.FacetCenters.x(i);
		ecent[i*N_CELLSPEREDGE+1] = sim.mesh.FacetCenters.y(i);

		eleng[i] = sim.mesh.FacetVolumes(i);

		//    	std::cout << "Edge " << i << "   left cell = " << sim.mesh.Facets[i].LeftCell() << "   right cell = " << sim.mesh.Facets[i].RightCell() << std::endl;
		//      std::cout << "Edge " << i << "   normal vector = [ " << enorm[i*N_CELLSPEREDGE] << " , " << enorm[i*N_CELLSPEREDGE+1] << " ]" << std::endl;
		//      std::cout << "Edge " << i << "   center vector = [ " << ecent[i*N_CELLSPEREDGE] << " , " << ecent[i*N_CELLSPEREDGE+1] << " ]" << std::endl;
		//      std::cout << "Edge " << i << "   length =  " << eleng[i] << std::endl;
	}

	std::cout << "Number of edges according to sim.mesh.Nodes.size() = " << sim.mesh.Facets.size() << std::endl;
	std::cout << "Number of edges according to ssim.mesh.NFaces      = " << sim.mesh.NFaces << "  <=== Accept this one! " << std::endl;

	//
	// Define OP2 sets
	//
	op_set nodes  = op_decl_set(nnode,  "nodes");
	op_set edges  = op_decl_set(nedge,  "edges");
	op_set cells  = op_decl_set(ncell,  "cells");

	//
	// Define OP2 set maps
	//
	op_map pcell   = op_decl_map(cells, nodes, N_NODESPERCELL, cell, "pcell");
	op_map pecell  = op_decl_map(edges, cells, N_CELLSPEREDGE, ecell, "pecell");
	op_map pccell  = op_decl_map(cells, cells, N_NODESPERCELL, ccell, "pccell");
	op_map pcedge  = op_decl_map(cells, edges, N_NODESPERCELL, cedge, "pcedge");

	//
	// Define OP2 datasets
	//
	op_dat p_ccent = op_decl_dat(cells, MESH_DIM, "double", ccent, "p_ccent");
	op_dat p_carea = op_decl_dat(cells, 1, "double", carea, "p_carea");
	op_dat p_enorm = op_decl_dat(edges, MESH_DIM, "double", enorm, "p_enorm");
	op_dat p_ecent = op_decl_dat(edges, MESH_DIM, "double", ecent, "p_ecent");
	op_dat p_eleng = op_decl_dat(edges, 1, "double", eleng, "p_eleng");
	op_dat p_x 	   = op_decl_dat(nodes, MESH_DIM, "double", x    ,"p_x");
	//op_dat p_w     = op_decl_dat(cells ,N_STATEVAR, "double",w    ,"p_w");
	//op_dat p_wold  = op_decl_dat(cells ,N_STATEVAR, "double",wold ,"p_wold");
	//op_dat p_res   = op_decl_dat(cells ,N_STATEVAR, "double",res  ,"p_res");

	//
	// Define HDF5 filename
	//
	char *filename_msh; // = "stlaurent_35k.msh";
	char *filename_h5; // = "stlaurent_35k.h5";
	filename_msh = strdup(sim.MeshFileName.c_str());
	filename_h5 = strndup(filename_msh, strlen(filename_msh)-4);
	strcat(filename_h5, ".h5");

	//
	// Write mesh and geometry data to HDF5
	//
	op_write_hdf5(filename_h5);

	//
	// Read constants and write to HDF5
	//
	double cfl = sim.CFL; // CFL condition
	op_write_const_hdf5("cfl", 1, "double", (char *)&cfl, filename_h5);
	double ftime = sim.FinalTime; // Final time: as defined by Volna the end of real-time simulation
	op_write_const_hdf5("ftime", 1, "double", (char *)&ftime, filename_h5);
	double dtmax = sim.Dtmax; // Maximum timestep
	op_write_const_hdf5("dtmax", 1, "double", (char *)&dtmax, filename_h5);
	double g = 9.81; // Gravity constant
	op_write_const_hdf5("g", 1, "double", (char *)&g, filename_h5);

	//WRITING VALUES MANUALLY
	hid_t file;
	herr_t status;
	file = H5Fopen(filename_h5, H5F_ACC_RDWR, H5P_DEFAULT);

	check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/x0", &sim.bore_params.x0));
	check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/Hl", &sim.bore_params.Hl));
	check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/ul", &sim.bore_params.ul));
	check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/vl", &sim.bore_params.vl));
	check_hdf5_error(H5LTmake_dataset_double(file, "BoreParams/S", &sim.bore_params.S));
	check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/A", &sim.gaussian_landslide_params.A));
	check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/v", &sim.gaussian_landslide_params.v));
	check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/lx", &sim.gaussian_landslide_params.lx));
	check_hdf5_error(H5LTmake_dataset_double(file, "GaussianLandslideParams/ly", &sim.gaussian_landslide_params.ly));

	//...
	check_hdf5_error(H5Fclose(file));

	free(cell);
	free(ecell);
	free(ccell);
	free(cedge);
	free(ccent);
	free(carea);
	free(enorm);
	free(ecent);
	free(eleng);
	free(x);
	//free(w);
	//free(wold);
	//free(res);

	op_exit();
}





//// op_mesh_io_import() function - Imports array data to GMSH mesh file, with *msh extension.
//// Arguments:
////   *filename 	- defines the filename where data is imported from
////	 mesh_dim 	- Mesh dimension. Note: separate from mesh node dimension (x,y,z)
////	 nodes_per_cell - No. of nodes per cell element
////
//// Returns:
//// 	 **x		- Node coordinates
////	 **cell 	- Cell node indices
////	 **edge 	- Edge node indices
////	 **ecell	- Edge to cell
////	 **bedge	- Boundary edge nodes
////	 **becell 	- Boundary edge to cell
//// 	 **bound	- Boundary indices
//// 	 *nnode		- No. of nodes in mesh
//// 	 *ncell		- No. of cells in mesh
//// 	 *nedge		- No. of edges collected by import
//// 	 *nbedge	- No. of boundary edges collected by import
////
//// Note: Detailed desciption of *.msh file format in GMSH User's Manual.
//inline void op_mesh_io_import(const char *filename, int mesh_dim, int nodes_per_cell, double **x, int **cell, int **edge, int **ecell, int **bedge, int **becell, int **bound, int *nnode, int *ncell, int *nedge, int *nbedge) {
//	/* For most users, a single LibMeshInit object should be created at
//	 * the start of your main() function.  This object replaces the
//	 * previous libMesh::init()/libMesh::close() methods, which are
//	 * now deprecated.
//	 */
//	libMesh::LibMeshInit libMeshInit();
//	// Create Mesh structure for holding mesh header-like data
//	libMesh::Mesh mesh(mesh_dim);
//	//libMesh::LibMeshInit init();
//	// Create MeshData structure to hold actual mesh data
//	//libMesh::MeshData mesh_data(mesh);
//	// Activate MeshData in order to be used for reading data
//	//mesh_data.activate();
//	// Read in grid
//	printf("Reading *.msh file...\n");
//	// Read the actual data. Note: Nodes, elements etc. are number from 0
//	GmshIO::GmshIO gmshio(mesh);
//	gmshio.read(filename);
//	// Argument true - skip renumbering nodes and elements
//	// prepare_forUse(): 3 steps: 1.) call find_neighbors() 2.) call partition() 3.) call renumber_nodes_and_elements()
//	mesh.prepare_for_use(true);
//	//mesh.read(filename,&mesh_data,true);
//	//libMesh::MetisPartitioner part;
//	//part.partition(mesh, 1);
//	// Prepare the newly created data for use
//	mesh.prepare_for_use();
//	printf("mesh.print_info(): \n");
//	mesh.print_info();
//	printf("mesh_data.print_info()t: \n");
//	//mesh_data.print_info();
//	printf("%s.msh file is successfully read.", filename);
//
//	*nnode = mesh.n_nodes();
//	*ncell = mesh.n_elem();
//
//	int abc[*nnode];
//
//	printf("\nCollecting data node coordinates and cell nodes...");
//	// Array storing node coordinates
//	*x		= (double*) malloc( mesh_dim * (*nnode) * sizeof(double));
//	// Array storing cells defined by nodes
//	*cell	= (int*) 	malloc( nodes_per_cell * (*ncell) * sizeof(int));
//	MeshBase::const_node_iterator node_it = mesh.nodes_begin();
//	const MeshBase::const_node_iterator node_it_end = mesh.nodes_end();
//	const Node* node = NULL;
//	int i;
//	for(i = 0; node_it != node_it_end; node_it++) {
//		node = *node_it;
//		//printf("\nID %d: ",node->id());
//		//node->print();
//		(*x)[i*mesh_dim  ] = (*node)(0);
//		(*x)[i*mesh_dim+1] = (*node)(1);
//		//printf("%d : ID %d : %f %f\n",i,node->id(),x[i*mesh_dim],x[i*mesh_dim+1]);
//		i++;
//	}
//	printf("done.\n");
//
//	printf("Collecting mapping data...\n");
//	mesh.find_neighbors();
//
//	// Store maps temporarily in a dynamically allocated datastructure - the number of edges is not known in advance
//	std::vector<int> edge_vec;
//	std::vector<int> ecell_vec;
//	std::vector<int> bedge_vec;
//	std::vector<int> becell_vec;
//	std::vector<int> bound_vec;
//
//	MeshBase::const_element_iterator elem_it = mesh.level_elements_begin(0);
//	const MeshBase::const_element_iterator elem_it_end = mesh.level_elements_end(0);
////	MeshBase::const_element_iterator elem_it = mesh.elements_begin();
////	const MeshBase::const_element_iterator elem_it_end = mesh.elements_end();
//	const libMesh::Elem* elem;
//	const libMesh::Elem* elem_n;
//
//	// Temporary node IDs by increasing index 1,2,3
//	int n0, n1, n2;
//	i = 0;
//	for(; elem_it != elem_it_end; ++elem_it) {
//		elem = *elem_it;
//		printf("Element ID %d \n", elem->id());
//		// Nodes of a cell - according to GMSH manual: indices 0,1,2 are in CW order
//		n0 = elem->get_node(0)->id();
//		n1 = elem->get_node(1)->id();
//		n2 = elem->get_node(2)->id();
//		// Store nodes of a cells
//		(*cell)[i*nodes_per_cell  ] = n0;
//		(*cell)[i*nodes_per_cell+1] = n1;
//		(*cell)[i*nodes_per_cell+2] = n2;
//		//printf("%d nodes: %d %d %d\n",i,n0,n1,n2);
//
//		// Getting edges from neighbouring cells
//		// http://geuz.org/gmsh/doc/texinfo/gmsh.html : 9.3.2 High order elements: ...Edges and faces are numbered following the lowest order template that generates a single high-order on this edge/face. Furthermore, an edge is oriented from the vertex with the lowest to the highest index. The orientation of a face is such that the computed normal points outward; the starting point is the vertex with the lowest index.
//		//printf("neighbor(%d,side) = ", elem->id());
//		// Iterate though the side of the cell
//		for(int side=0; side<elem->n_sides(); side++) {
//			// If a neighbor of a cell exists save the nodes of the between the neighboring cells
//			if(elem->neighbor(side)!=NULL) {
//				elem_n = elem->neighbor(side);
//				//printf("%d ", elem_n->id());
//				// Store nodes of an edge only once. This condition assures that are stored on once.
//				if(elem->id() < elem_n->id())	{
//					if(side==0) {
//						edge_vec.push_back(n0);
//						edge_vec.push_back(n1);
//					}
//					if(side==1) {
//						edge_vec.push_back(n1);
//						edge_vec.push_back(n2);
//					}
//					if(side==2) {
//						edge_vec.push_back(n2);
//						edge_vec.push_back(n0);
//					}
//					// Store that the cell IDs which are on the two sides of the edge
//					ecell_vec.push_back(elem->id());
//					ecell_vec.push_back(elem_n->id());
//					// Keep track of the number of stored edge-node and edge-cell maps
//					(*nedge)++;
//					//printf(" nedge = %d\n",nedge);
//				}
//			// If a cell doesn't have a neighbor on a side, store that edge as a boundary edge
//			} else {
//				const std::vector<boundary_id_type>& side_boundaries = mesh.boundary_info->boundary_ids(elem,side);
//				int tmp = mesh.boundary_info->boundary_id(elem,side);
//				// If no boundary ID is present in the vector OR Boundary ID is -1234 (invalid boundary ID) report the fact
//				if(side_boundaries.size()==0 || tmp == -1234) {
//					printf("No boundary ID defined on element %d on edge %d \n", elem->id(), side);
//					printf("Exiting process. No output is prepared. \n");
//					exit(-1);
//				} else {
//					if(side==0) {
//						bedge_vec.push_back(n0);
//						bedge_vec.push_back(n1);
//					}
//					if(side==1) {
//						bedge_vec.push_back(n1);
//						//printf(" nedge = %d\n",nedge);
//						bedge_vec.push_back(n2);
//					}
//					if(side==2) {
//						bedge_vec.push_back(n2);
//						bedge_vec.push_back(n0);
//					}
//					// Store that one cell ID which is on the edge
//					becell_vec.push_back(elem->id());
//					bound_vec.push_back(side_boundaries[0]);
//					//bound_vec.push_back(mesh.boundary_info->boundary_id(elem,side)); // Deprecated
//					// Keep track of the number of stored boundaryedge-node, boundaryedge-cell and boundary maps
//					(*nbedge)++;
//				}
//
//			}
//		}
//		//printf("\n");
//		//printf("(%d,side) = ", elem->id());
//
//		// Getting boundaries
//		//printf("boundary(%d,side) = ", elem->id());
//		//for(int side=0; side<elem->n_sides(); side++) {
//		//		printf("%d ", mesh.boundary_info->boundary_id(elem,side));
//		//}
//		//printf("\n");
//		i++;
//	}
//

//
//	// Copy data from dynamic STL vector to static array
//	for(int i=0; i< (*nedge); i++) {
//		(*edge) [2*i  ] = edge_vec [2*i  ];
//		(*edge) [2*i+1] = edge_vec [2*i+1];
//		(*ecell)[2*i  ] = ecell_vec[2*i  ];
//		(*ecell)[2*i+1] = ecell_vec[2*i+1];
//		//printf("Edge node map %d : %d %d \n", i, (*edge)[2*i], (*edge)[2*i+1]);
//		//printf("Edge cell map %d : %d %d \n", i, (*ecell)[2*i], (*ecell)[2*i+1]);
//	}
//
//	for(int i=0; i< (*nbedge); i++) {
//		(*bedge) [2*i  ] = bedge_vec  [2*i  ];
//		(*bedge) [2*i+1] = bedge_vec  [2*i+1];
//		(*becell)[i    ] = becell_vec [i    ];
//		(*bound) [i    ] = bound_vec  [i    ];
//		//printf("Boundary Edge-node map %d : %d %d \n", i, (*bedge)[2*i], (*bedge)[2*i+1]);
//		//printf("Boundary Edge-cell map %d : %d 	 \n", i, (*becell)[i]);
//		//printf("Boundary ID            %d : %d 	 \n", i, (*bound)[i]);
//	}
//}
//
//
//
//// op_mesh_io_export() function - Exports array data to GMSH mesh file, with *msh extension.
//// Arguments:
////   dat      - OP dataset ID to get flow variable data from
////   flow_var - index of flow variable to be exported from dat
////   filename - defines the filename where data is exported
////
//// Returns:
////   0     - if data export was succesfull
////   other - if error occured during export
////
//// Note: 2 dimensional mesh is used. Detailed desciption of *.msh file format in GMSH User's Manual.
////int op_mesh_io_export(op_dat dat, int flow_var, char* filename, double* x, int nnode, int* cell, int ncell) {
//void op_mesh_io_export(double* q, int flow_var, char* filename, int nodes_per_cell, double* x, int nnode, int* cell, int ncell) {
//  //op_fetch_data(dat);
//  printf("Exporting data...");
//  FILE *msh_fp;
//  if ( (msh_fp = fopen(filename,"w")) == NULL) {
//    printf("\ncan't write file *.msh\n"); exit(-1);
//  }
//  // MeshFormat: version-number file-type data-size
//  fprintf(msh_fp,"$MeshFormat \n2.0 0 8 \n$EndMeshFormat\n");
//  fprintf(msh_fp,"$Nodes \n%i \n",nnode);
//  // Node IDs start from 1 in GMSH
//  for(int i=0; i<nnode; i++) fprintf(msh_fp,"%i %lf %lf %lf \n",i+1, x[i*2], x[i*2+1], 0.0);
//  fprintf(msh_fp,"$EndNodes \n");
//  fprintf(msh_fp,"$Elements \n%i \n",ncell);
//  // Element & node IDs start from 1 in GMSH
//  for(int i=0; i<ncell; i++) fprintf(msh_fp,"%i 2 0 %i %i %i \n",i+1,cell[i*nodes_per_cell]+1,cell[i*nodes_per_cell+1]+1,cell[i*nodes_per_cell+2]+1);
//  fprintf(msh_fp,"$EndElements \n");
//  fprintf(msh_fp,"$ElementData \n1 \n\"Density view\" \n1 \n0.0 \n3 \n0 \n1 \n%i \n",ncell);
//  // Print scalar values for each element
//  for(int i=0; i<ncell; i++) fprintf(msh_fp,"%i %1.10lf \n",i+1,q[i*4+flow_var]);
//  fprintf(msh_fp,"$EndElementData \n");
//  fclose(msh_fp);
//  printf("done.\n");
//}
//
////
//// main program
////
//





//	op_exit();
//}
