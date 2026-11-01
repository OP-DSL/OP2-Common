#pragma once

// A mesh file's nodes and edges, read by each rank in blocks: MG-CFD's
// (edge-->node, numbered from 1, and node_coordinates) or airfoil's (pedge
// and p_x).

#include <vector>

struct MeshBlock {
    int nodes = 0, edges = 0;               // in the whole mesh
    int n0 = 0, n1 = 0, e0 = 0, e1 = 0;     // this rank's blocks of them
    int dim = 0;
    std::vector<double> coords;             // (n1 - n0) * dim
    std::vector<int> edge_nodes;            // (e1 - e0) * 2, numbered from 0
};

// Stops the run, saying why, if path is no such mesh.
MeshBlock read_mesh(const char *path, int rank, int ranks);
