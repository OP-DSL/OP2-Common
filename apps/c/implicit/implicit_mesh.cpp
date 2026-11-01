#include "implicit_mesh.h"

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

#ifdef IMPLICIT_HDF5
#include <hdf5.h>

namespace {

[[noreturn]] void fail(const char *path, const char *what) {
    fprintf(stderr, "implicit: mesh %s: %s\n", path, what);
    exit(1);
}

// Rows [row0, row0 + rows) of a two-dimensional dataset, cols wide.
template<typename T>
std::vector<T> read_rows(const char *path, hid_t file, const char *name, hid_t type, hsize_t row0,
                         hsize_t rows, hsize_t cols) {
    hid_t set = H5Dopen(file, name, H5P_DEFAULT);
    if (set < 0)
        fail(path, name);
    hid_t space = H5Dget_space(set);
    hsize_t start[2] = {row0, 0}, count[2] = {rows, cols};
    H5Sselect_hyperslab(space, H5S_SELECT_SET, start, NULL, count, NULL);
    hid_t memory = H5Screate_simple(2, count, NULL);
    std::vector<T> out(rows * cols);
    if (H5Dread(set, type, memory, space, H5P_DEFAULT, out.data()) < 0)
        fail(path, name);
    H5Sclose(memory);
    H5Sclose(space);
    H5Dclose(set);
    return out;
}

void extent(const char *path, hid_t file, const char *name, hsize_t dims[2]) {
    hid_t set = H5Dopen(file, name, H5P_DEFAULT);
    if (set < 0)
        fail(path, name);
    hid_t space = H5Dget_space(set);
    if (H5Sget_simple_extent_ndims(space) != 2)
        fail(path, "a mesh dataset is not two-dimensional");
    H5Sget_simple_extent_dims(space, dims, NULL);
    H5Sclose(space);
    H5Dclose(set);
}

} // namespace

MeshBlock read_mesh(const char *path, int rank, int ranks) {
    hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0)
        fail(path, "cannot open it");
    const bool mgcfd = H5Lexists(file, "edge-->node", H5P_DEFAULT) > 0;
    if (!mgcfd && H5Lexists(file, "pedge", H5P_DEFAULT) <= 0)
        fail(path, "neither an MG-CFD mesh (edge-->node) nor an airfoil one (pedge)");
    // Airfoil's boundary edges follow its interior ones, so every node has one.
    const char *edges = mgcfd ? "edge-->node" : "pedge", *more = mgcfd ? NULL : "pbedge";
    const char *coords = mgcfd ? "node_coordinates" : "p_x";
    const int base = mgcfd ? 1 : 0;

    hsize_t edims[2], mdims[2] = {0, 2}, cdims[2];
    extent(path, file, edges, edims);
    if (more != NULL)
        extent(path, file, more, mdims);
    extent(path, file, coords, cdims);
    if (edims[1] != 2 || mdims[1] != 2 || (cdims[1] != 2 && cdims[1] != 3))
        fail(path, "edges must join two nodes, and coordinates be 2D or 3D");

    MeshBlock m;
    m.nodes = (int)cdims[0];
    m.edges = (int)(edims[0] + mdims[0]);
    m.dim = (int)cdims[1];
    auto block = [&](int total, int r) { return (int)((long long)total * r / ranks); };
    m.n0 = block(m.nodes, rank);
    m.n1 = block(m.nodes, rank + 1);
    m.e0 = block(m.edges, rank);
    m.e1 = block(m.edges, rank + 1);
    m.coords = read_rows<double>(path, file, coords, H5T_NATIVE_DOUBLE, m.n0, m.n1 - m.n0, m.dim);
    const int first = (int)edims[0];
    if (m.e0 < first)
        m.edge_nodes = read_rows<int>(path, file, edges, H5T_NATIVE_INT, m.e0, std::min(m.e1, first) - m.e0, 2);
    if (m.e1 > first) {
        const int from = std::max(m.e0, first) - first;
        std::vector<int> rest = read_rows<int>(path, file, more, H5T_NATIVE_INT, from, m.e1 - first - from, 2);
        m.edge_nodes.insert(m.edge_nodes.end(), rest.begin(), rest.end());
    }
    H5Fclose(file);

    for (int& node : m.edge_nodes) {
        node -= base;
        if (node < 0 || node >= m.nodes)
            fail(path, "an edge names a node outside the mesh");
    }
    return m;
}
#else
MeshBlock read_mesh(const char *path, int, int) {
    fprintf(stderr, "implicit: mesh %s: this build has no HDF5\n", path);
    exit(1);
}
#endif
