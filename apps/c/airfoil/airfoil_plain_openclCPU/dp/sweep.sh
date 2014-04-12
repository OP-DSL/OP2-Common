#!/bin/bash
# Set error checking: if a statment returns other than 0, the script stops executing
set +e 
set -u

# Use this tool to sweep a problem space

if [[ -z "$1" ]] ; then
	echo "Specify an Airfoil implementation!"
	exit 
fi

echo "Sweeping started..."

# Create directories for logging and benchmarking
mkdir -p log
mkdir -p benchmark

# Set iteration number and precision
export BINARY=$1

# Set sweep data file
SWEEPTXT="sweep_$1.txt"
rm -f $SWEEPTXT

# Set environment variables to use thread pinning
export KMP_AFFINITY=compact,0

# Sweep through problem N^3 size
for N in 128 256 512 768 1024 1536 2048 3072 4096 
do
  echo $N
  # Execute simulation to measure total execution time
  #./$BINARY new_grid.dat OP_PART_SIZE=$N OP_BLOCK_SIZE=$N 2>&1 | tee -a $SWEEPTXT
  #KMP_AFFINITY=compact OMP_NUM_THREADS=12 numactl --cpunodebind=1 ./$BINARY OP_PART_SIZE=$N 2>&1 | tee -a $SWEEPTXT
  KMP_AFFINITY=compact OMP_NUM_THREADS=12 mpirun -np 1 ./numawrap2 ./$BINARY OP_PART_SIZE=$N 2>&1 | tee -a $SWEEPTXT
  #KMP_AFFINITY=compact OMP_NUM_THREADS=16 numactl --cpunodebind=1 ./$BINARY OP_PART_SIZE=$N
done

echo "Sweeping ended.\n"

# Turn off error checking
set -e
