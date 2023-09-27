import h5py
import pyvista as pv
import numpy as np

# Read your mesh file which contains physical coordinates of each cell
with h5py.File('FE_grid.h5', 'r') as f:
    # Assuming your data is stored in datasets 'x', 'y', 'z'
    x = f['p_x'][:]

x = np.column_stack((x, np.zeros(x.shape[0])))


# Loop over your data files
for i in range(73):
    with h5py.File(f'data/h5s/aero_n100_m100_c0.100000_ts{i+1}.h5', 'r') as f:
        # Assuming your data is stored in a dataset 'data'
        data = f['p_phim'][:]
        print(f'reading file {i+1}')
    # Add your data to the grid
    
    grid = pv.UnstructuredGrid()
    grid.points = x

    grid.point_data[f'p_phim_{i}'] = np.array(data)
    grid.save(f'data/vtks/simulation_data_{i}.vtk', binary=False)
    print(f'data/vtks/simulation_data_{i}.vtk generated')

# Write the data to a .vtk file
#grid.save('data/vtks/simulation_data.vtk', binary=False)