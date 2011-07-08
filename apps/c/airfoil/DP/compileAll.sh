

for PART_SIZE_ENV in 64 128 256 512
do
	for BLOCK_SIZE_ENV in 64 128 256 512
	do

	  make clean
	  make airfoil_cuda

	done
done
