make airfoil_cuda airfoil_opencl
./airfoil_cuda > cuda.log
./airfoil_opencl > opencl.log
meld cuda.log opencl.log

