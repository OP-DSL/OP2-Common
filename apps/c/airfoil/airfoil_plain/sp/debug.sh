make airfoil_cuda airfoil_opencl
./airfoil_cuda OP_PART_SIZE=32 OP_BLOCK_SIZE=32 
mv out.txt out_cuda.txt
./airfoil_opencl OP_PART_SIZE=32 OP_BLOCK_SIZE=32 
mv out.txt out_opencl.txt
#meld out_cuda.txt out_opencl.txt
vimdiff out_cuda.txt out_opencl.txt

