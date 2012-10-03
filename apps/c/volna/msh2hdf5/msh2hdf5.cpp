#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<vector>
#include<string.h>

//
// libMesh headers
//

#include"libmesh.h"
#include"mesh.h"
#include"elem.h"
#include"mesh_data.h"
#include"boundary_info.h"
#include"gmsh_io.h"

//
// OP header file
//

//#include "op_lib_cpp.h"
//#include "op_lib_mpi.h"
//#include "op_util.h"

//
// op_par_loop declarations
//

#include "op_seq.h"

// op_mesh_io_import() function - Imports array data to GMSH mesh file, with *msh extension.
// Arguments:
//   *filename 	- defines the filename where data is imported from
//	 mesh_dim 	- Mesh dimension. Note: separate from mesh node dimension (x,y,z)
//	 nodes_per_cell - No. of nodes per cell element
//
// Returns:
// 	 **x		- Node coordinates
//	 **cell 	- Cell node indices
//	 **edge 	- Edge node indices
//	 **ecell	- Edge to cell
//	 **bedge	- Boundary edge nodes
//	 **becell 	- Boundary edge to cell
// 	 **bound	- Boundary indices
// 	 *nnode		- No. of nodes in mesh
// 	 *ncell		- No. of cells in mesh
// 	 *nedge		- No. of edges collected by import
// 	 *nbedge	- No. of boundary edges collected by import
//
// Note: Detailed desciption of *.msh file format in GMSH User's Manual.
inline void op_mesh_io_import(const char *filename, int mesh_dim, int nodes_per_cell, double **x, int **cell, int **edge, int **ecell, int **bedge, int **becell, int **bound, int *nnode, int *ncell, int *nedge, int *nbedge) {
	/* For most users, a single LibMeshInit object should be created at
	 * the start of your main() function.  This object replaces the
	 * previous libMesh::init()/libMesh::close() methods, which are
	 * now deprecated.
	 */
	libMesh::LibMeshInit libMeshInit();
	// Create Mesh structure for holding mesh header-like data
	libMesh::Mesh mesh(mesh_dim);
	//libMesh::LibMeshInit init();
	// Create MeshData structure to hold actual mesh data
	//libMesh::MeshData mesh_data(mesh);
	// Activate MeshData in order to be used for reading data
	//mesh_data.activate();
	// Read in grid
	printf("Reading *.msh file...\n");
	// Read the actual data. Note: Nodes, elements etc. are number from 0
	GmshIO::GmshIO gmshio(mesh);
	gmshio.read(filename);
	// Argument true - skip renumbering nodes and elements
	// prepare_forUse(): 3 steps: 1.) call find_neighbors() 2.) call partition() 3.) call renumber_nodes_and_elements()
	mesh.prepare_for_use(true);
	//mesh.read(filename,&mesh_data,true);
	//libMesh::MetisPartitioner part;
	//part.partition(mesh, 1);
	// Prepare the newly created data for use
	mesh.prepare_for_use();
	printf("mesh.print_info(): \n");
	mesh.print_info();
	printf("mesh_data.print_info()t: \n");
	//mesh_data.print_info();
	printf("%s.msh file is successfully read.", filename);

	*nnode = mesh.n_nodes();
	*ncell = mesh.n_elem();

	int abc[*nnode];

	printf("\nCollecting data node coordinates and cell nodes...");
	// Array storing node coordinates
	*x		= (double*) malloc( mesh_dim * (*nnode) * sizeof(double));
	// Array storing cells defined by nodes
	*cell	= (int*) 	malloc( nodes_per_cell * (*ncell) * sizeof(int));
	MeshBase::const_node_iterator node_it = mesh.nodes_begin();
	const MeshBase::const_node_iterator node_it_end = mesh.nodes_end();
	const Node* node = NULL;
	int i;
	for(i = 0; node_it != node_it_end; node_it++) {
		node = *node_it;
		//printf("\nID %d: ",node->id());
		//node->print();
		(*x)[i*mesh_dim  ] = (*node)(0);
		(*x)[i*mesh_dim+1] = (*node)(1);
		//printf("%d : ID %d : %f %f\n",i,node->id(),x[i*mesh_dim],x[i*mesh_dim+1]);
		i++;
	}
	printf("done.\n");

	printf("Collecting mapping data...\n");
	mesh.find_neighbors();

	// Store maps temporarily in a dynamically allocated datastructure - the number of edges is not known in advance
	std::vector<int> edge_vec;
	std::vector<int> ecell_vec;
	std::vector<int> bedge_vec;
	std::vector<int> becell_vec;
	std::vector<int> bound_vec;

	MeshBase::const_element_iterator elem_it = mesh.level_elements_begin(0);
	const MeshBase::const_element_iterator elem_it_end = mesh.level_elements_end(0);
//	MeshBase::const_element_iterator elem_it = mesh.elements_begin();
//	const MeshBase::const_element_iterator elem_it_end = mesh.elements_end();
	const libMesh::Elem* elem;
	const libMesh::Elem* elem_n;

	// Temporary node IDs by increasing index 1,2,3
	int n0, n1, n2;
	i = 0;
	for(; elem_it != elem_it_end; ++elem_it) {
		elem = *elem_it;
		printf("Element ID %d \n", elem->id());
		// Nodes of a cell - according to GMSH manual: indices 0,1,2 are in CW order
		n0 = elem->get_node(0)->id();
		n1 = elem->get_node(1)->id();
		n2 = elem->get_node(2)->id();
		// Store nodes of a cells
		(*cell)[i*nodes_per_cell  ] = n0;
		(*cell)[i*nodes_per_cell+1] = n1;
		(*cell)[i*nodes_per_cell+2] = n2;
		//printf("%d nodes: %d %d %d\n",i,n0,n1,n2);

		// Getting edges from neighbouring cells
		// http://geuz.org/gmsh/doc/texinfo/gmsh.html : 9.3.2 High order elements: ...Edges and faces are numbered following the lowest order template that generates a single high-order on this edge/face. Furthermore, an edge is oriented from the vertex with the lowest to the highest index. The orientation of a face is such that the computed normal points outward; the starting point is the vertex with the lowest index.
		//printf("neighbor(%d,side) = ", elem->id());
		// Iterate though the side of the cell
		for(int side=0; side<elem->n_sides(); side++) {
			// If a neighbor of a cell exists save the nodes of the between the neighboring cells
			if(elem->neighbor(side)!=NULL) {
				elem_n = elem->neighbor(side);
				//printf("%d ", elem_n->id());
				// Store nodes of an edge only once. This condition assures that are stored on once.
				if(elem->id() < elem_n->id())	{
					if(side==0) {
						edge_vec.push_back(n0);
						edge_vec.push_back(n1);
					}
					if(side==1) {
						edge_vec.push_back(n1);
						edge_vec.push_back(n2);
					}
					if(side==2) {
						edge_vec.push_back(n2);
						edge_vec.push_back(n0);
					}
					// Store that the cell IDs which are on the two sides of the edge
					ecell_vec.push_back(elem->id());
					ecell_vec.push_back(elem_n->id());
					// Keep track of the number of stored edge-node and edge-cell maps
					(*nedge)++;
					//printf(" nedge = %d\n",nedge);
				}
			// If a cell doesn't have a neighbor on a side, store that edge as a boundary edge
			} else {
				const std::vector<boundary_id_type>& side_boundaries = mesh.boundary_info->boundary_ids(elem,side);
				int tmp = mesh.boundary_info->boundary_id(elem,side);
				// If no boundary ID is present in the vector OR Boundary ID is -1234 (invalid boundary ID) report the fact
				if(side_boundaries.size()==0 || tmp == -1234) {
					printf("No boundary ID defined on element %d on edge %d \n", elem->id(), side);
					printf("Exiting process. No output is prepared. \n");
					exit(-1);
				} else {
					if(side==0) {
						bedge_vec.push_back(n0);
						bedge_vec.push_back(n1);
					}
					if(side==1) {
						bedge_vec.push_back(n1);
						//printf(" nedge = %d\n",nedge);
						bedge_vec.push_back(n2);
					}
					if(side==2) {
						bedge_vec.push_back(n2);
						bedge_vec.push_back(n0);
					}
					// Store that one cell ID which is on the edge
					becell_vec.push_back(elem->id());
					bound_vec.push_back(side_boundaries[0]);
					//bound_vec.push_back(mesh.boundary_info->boundary_id(elem,side)); // Deprecated
					// Keep track of the number of stored boundaryedge-node, boundaryedge-cell and boundary maps
					(*nbedge)++;
				}

			}
		}
		//printf("\n");
		//printf("(%d,side) = ", elem->id());

		// Getting boundaries
		//printf("boundary(%d,side) = ", elem->id());
		//for(int side=0; side<elem->n_sides(); side++) {
		//		printf("%d ", mesh.boundary_info->boundary_id(elem,side));
		//}
		//printf("\n");
		i++;
	}

	// Arrays for mapping data
	*edge	= (int*) malloc(2* (*nedge)  *sizeof(int));
	*ecell	= (int*) malloc(2* (*nedge)  *sizeof(int));
	*bedge	= (int*) malloc(2* (*nbedge) *sizeof(int));
	*becell	= (int*) malloc(   (*nbedge) *sizeof(int));
	*bound	= (int*) malloc(   (*nbedge) *sizeof(int));

	// Copy data from dynamic STL vector to static array
	for(int i=0; i< (*nedge); i++) {
		(*edge) [2*i  ] = edge_vec [2*i  ];
		(*edge) [2*i+1] = edge_vec [2*i+1];
		(*ecell)[2*i  ] = ecell_vec[2*i  ];
		(*ecell)[2*i+1] = ecell_vec[2*i+1];
		//printf("Edge node map %d : %d %d \n", i, (*edge)[2*i], (*edge)[2*i+1]);
		//printf("Edge cell map %d : %d %d \n", i, (*ecell)[2*i], (*ecell)[2*i+1]);
	}

	for(int i=0; i< (*nbedge); i++) {
		(*bedge) [2*i  ] = bedge_vec  [2*i  ];
		(*bedge) [2*i+1] = bedge_vec  [2*i+1];
		(*becell)[i    ] = becell_vec [i    ];
		(*bound) [i    ] = bound_vec  [i    ];
		//printf("Boundary Edge-node map %d : %d %d \n", i, (*bedge)[2*i], (*bedge)[2*i+1]);
		//printf("Boundary Edge-cell map %d : %d 	 \n", i, (*becell)[i]);
		//printf("Boundary ID            %d : %d 	 \n", i, (*bound)[i]);
	}
}



// op_mesh_io_export() function - Exports array data to GMSH mesh file, with *msh extension.
// Arguments:
//   dat      - OP dataset ID to get flow variable data from
//   flow_var - index of flow variable to be exported from dat
//   filename - defines the filename where data is exported
//
// Returns:
//   0     - if data export was succesfull
//   other - if error occured during export
//
// Note: 2 dimensional mesh is used. Detailed desciption of *.msh file format in GMSH User's Manual.
//int op_mesh_io_export(op_dat dat, int flow_var, char* filename, double* x, int nnode, int* cell, int ncell) {
void op_mesh_io_export(double* q, int flow_var, char* filename, int nodes_per_cell, double* x, int nnode, int* cell, int ncell) {
  //op_fetch_data(dat);
  printf("Exporting data...");
  FILE *msh_fp;
  if ( (msh_fp = fopen(filename,"w")) == NULL) {
    printf("\ncan't write file *.msh\n"); exit(-1);
  }
  // MeshFormat: version-number file-type data-size
  fprintf(msh_fp,"$MeshFormat \n2.0 0 8 \n$EndMeshFormat\n");
  fprintf(msh_fp,"$Nodes \n%i \n",nnode);
  // Node IDs start from 1 in GMSH
  for(int i=0; i<nnode; i++) fprintf(msh_fp,"%i %lf %lf %lf \n",i+1, x[i*2], x[i*2+1], 0.0);
  fprintf(msh_fp,"$EndNodes \n");
  fprintf(msh_fp,"$Elements \n%i \n",ncell);
  // Element & node IDs start from 1 in GMSH
  for(int i=0; i<ncell; i++) fprintf(msh_fp,"%i 2 0 %i %i %i \n",i+1,cell[i*nodes_per_cell]+1,cell[i*nodes_per_cell+1]+1,cell[i*nodes_per_cell+2]+1);
  fprintf(msh_fp,"$EndElements \n");
  fprintf(msh_fp,"$ElementData \n1 \n\"Density view\" \n1 \n0.0 \n3 \n0 \n1 \n%i \n",ncell);
  // Print scalar values for each element
  for(int i=0; i<ncell; i++) fprintf(msh_fp,"%i %1.10lf \n",i+1,q[i*4+flow_var]);
  fprintf(msh_fp,"$EndElementData \n");
  fclose(msh_fp);
  printf("done.\n");
}

//
// main program
//

int main(int argc, char **argv) {
	if(argc != 2) {
		printf("Wrong parameters! Please specify the GMSH filename with or without .msh extension, e.g. ./msh2hdf5 this_is_a_mesh \n");
		exit(-1);
	}

	op_printf("Initializing OP2...\n");
	op_init(argc, argv, 2);

	int *cell=NULL, *edge=NULL, *ecell=NULL, *bedge=NULL, *becell=NULL, *bound=NULL;
	double *x=NULL, *q=NULL, *qold=NULL, *res=NULL, *adt=NULL;

	int nnode=0, ncell=0, nedge=0, nbedge=0, niter=0;

	/**------------------------BEGIN  I/O -------------------**/
	char* filename = argv[1];
	char filename_msh[255]; // = "stlaurent_35k_ASCII.msh";
	char filename_h5[255]; // = "stlaurent_35k_ASCII.h5";
	char* p = strstr(filename,".msh");
	if(p==NULL) {
		strcpy(filename_msh, filename);
		strcpy(filename_h5, filename);
		strcat(filename_msh, ".msh");
		strcat(filename_h5, ".h5");
	} else {
		strcpy(filename_msh, filename);
		strncpy(filename_h5, filename, strlen(filename)-4);
		strcat(filename_h5, ".h5");
	}


	/* read in grid from disk on root processor */
	printf("Importing data from *.msh file...\n");
	op_mesh_io_import(filename_msh, 2, 3, &x, &cell, &edge, &ecell, &bedge, &becell, &bound, &nnode, &ncell, &nedge, &nbedge);

	printf("GMSH file data statistics: \n");
	printf("  No. nodes = %d\n",nnode);
	printf("  No. cells = %d\n",ncell);
	printf("Collected data statistics: \n");
	printf("  No. of edges           = %d\n", nedge);
	printf("  No. of boundary edges  = %d\n\n", nbedge);

	x      = (double *) malloc(2*nnode*sizeof(double));
	q      = (double *) malloc(4*ncell*sizeof(double));
	qold   = (double *) malloc(4*ncell*sizeof(double));
	res    = (double *) malloc(4*ncell*sizeof(double));
	adt    = (double *) malloc(  ncell*sizeof(double));

	op_set nodes  = op_decl_set(nnode,  "nodes");
	op_set edges  = op_decl_set(nedge,  "edges");
	op_set bedges = op_decl_set(nbedge, "bedges");
	op_set cells  = op_decl_set(ncell,  "cells");

	op_map pedge   = op_decl_map(edges, nodes,2,edge,  "pedge");
	op_map pecell  = op_decl_map(edges, cells,2,ecell, "pecell");
	op_map pbedge  = op_decl_map(bedges,nodes,2,bedge, "pbedge");
	op_map pbecell = op_decl_map(bedges,cells,1,becell,"pbecell");
	op_map pcell   = op_decl_map(cells, nodes,3,cell,  "pcell");

	op_dat p_bound = op_decl_dat(bedges,1,"int"  ,bound,"p_bound");
	op_dat p_x     = op_decl_dat(nodes ,2,"double",x    ,"p_x");
	op_dat p_q     = op_decl_dat(cells ,4,"double",q    ,"p_q");
	op_dat p_qold  = op_decl_dat(cells ,4,"double",qold ,"p_qold");
	op_dat p_adt   = op_decl_dat(cells ,1,"double",adt  ,"p_adt");
	op_dat p_res   = op_decl_dat(cells ,4,"double",res  ,"p_res");

	op_write_hdf5(filename_h5);

	op_exit();
}




















////	printf("Number of edges: %d\n", nedge);
////	for(int i=0; i<nedge; i++) {
////		printf("  %d: %d %d\n", i, edge_vec[i*2], edge_vec[i*2+1]);
////	}
//
//	// Init flow and temporary variables
//	double* q = (double*) malloc(4*ncell*sizeof(double));
//	for(int i=0; i<4*ncell; i++)
//		q[i] = 0.0;
//
//	op_mesh_io_export(q, 0, "./stlaurent_35k_ASCII_result.msh", 3, x, nnode, cell, ncell);
//
//	free(x);
//	free(cell);
//	free(edge);
//	free(ecell);
//	free(bedge);
//	free(becell);
//	free(bound);
//
//	return(0);
//}
