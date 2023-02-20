#!/bin/bash

PREC1="sp"
T_TOTAL="180000"
PREC2="dp"

#P_q_REF="/home/sikba/OP2_mixed/OP2-Common/apps/c/airfoil/airfoil_hdf5/qp/p_q-quad_2.8m_1000it.h5-start_in_sp"
P_q_REF="/home/sikba/OP2_mixed/OP2-Common/apps/c/airfoil/airfoil_hdf5/qp/p_q-quad_2.8m_180kit.h5-start_in_sp"

O1=$(pwd)/o1
O2=$(pwd)/o2
O3=$(pwd)/o3

PREC1L=""
if [[ ${PREC1} == "sp" ]]
then
  PREC1L="single"
elif [[ ${PREC1} == "dp" ]]
then
  PREC1L="double"
elif [[ ${PREC1} == "qp" ]]
then
  echo "CUDA can't handle long double"
  exit
  PREC1L="quad"
fi

PREC2L=""
if [[ $PREC2 == "sp" ]]
then
  PREC2L="single"
elif [[ $PREC2 == "dp" ]]
then
  PREC2L="double"
elif [[ $PREC2 == "qp" ]]
then
  echo "CUDA can't handle long double"
  exit
  PREC2L="quad"
fi

if [[ "$PREC2" != "no" ]]; 
then
  echo "Running auto mixed precision Airfoil, with precision $PREC1L and then $PREC2L, with $T_TOTAL iteration together. "
else
  echo "Running only $PREC1L precision Airfoil, with $T_TOTAL iteration."
fi

cd ../$PREC1
echo "Running in $PREC1L precision"
#mpirun -np 2 ./airfoil_mpi_cuda -i $T_TOTAL -o p_q_2.8m-${PREC1L}_it${T_TOTAL}.h5 > $O1

LOW_IT=$(cat $O1 | sed -nr 's/Breaking at iteration (.*)/\1/p')
#mv p_q_2.8m-${PREC1L}_it${T_TOTAL}.h5 p_q_2.8m-${PREC1L}_it${LOW_IT}.h5
HIGH_IT=$(($T_TOTAL-$LOW_IT))

echo "$PREC1L converged at iteration $LOW_IT, continuing with $PREC2L for $HIGH_IT iterations."

if [[ "$PREC2" != "no" ]]; 
then
  cd ../$PREC2
  echo "Running in $PREC2L precision"
 # mpirun -np 2 --oversubscribe ~/numawrap ./airfoil_mpi_cuda -i $HIGH_IT -d ../${PREC1}/p_q_2.8m-${PREC1L}_it${LOW_IT}.h5 -o p_q_2.8m-${LOW_IT}${PREC1L}_${HIGH_IT}${PREC2L}.h5 > $O2
fi

cd ../compr
echo "Comparing results"
if [[ "$PREC2" != "no" ]]; 
then
  mpirun -np 1 --oversubscribe ~/numawrap ./airfoil_mpi_genseq -A ../${PREC2}/p_q_2.8m-${LOW_IT}${PREC1L}_${HIGH_IT}${PREC2L}.h5 -B ${P_q_REF} -o airfoil_reldiff_p_q_2.8m-${LOW_IT}${PREC1}_${HIGH_IT}${PREC2}.vtk > $O3
else
  mpirun -np 1 --oversubscribe ~/numawrap ./airfoil_mpi_genseq -A ../${PREC1}/p_q_2.8m-${PREC1L}_it${LOW_IT}.h5 -B ${P_q_REF} -o airfoil_reldiff_p_q_2.8m-${LOW_IT}${PREC1}}.vtk > $O3
fi

MTR1=$(cat $O1 | sed -nr 's/Max total runtime = (.*)/\1/p')
MTR2=$(cat $O2 | sed -nr 's/Max total runtime = (.*)/\1/p')
SMTR=$(awk "BEGIN{ print ${MTR1} + ${MTR2} }")
echo "Sum max total runtime: ${SMTR}"
cat $O3 | tail -n 6 | head -n 4

if [[ "$PREC2" != "no" ]]; 
then
  echo "Relative difference mesh results in: $(pwd)/airfoil_reldiff_p_q_2.8m-${LOW_IT}${PREC1}_${HIGH_IT}${PREC2}.vtk"
else
  echo "Relative difference mesh results in: $(pwd)/airfoil_reldiff_p_q_2.8m-${LOW_IT}${PREC1}.vtk"
fi
