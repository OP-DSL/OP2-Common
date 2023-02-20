#!/bin/bash

PREC1="sp"
T1="900"
PREC2="dp"
T2=$((1000-$T1))

P_q_REF="/home/sikba/OP2_mixed/OP2-Common/apps/c/airfoil/airfoil_hdf5/qp/p_q-quad_2.8m_1000it.h5-start_in_sp"

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
  PREC2L="quad"
fi

if [[ "$PREC2" != "no" ]]; 
then
  echo "Running mixed precision Airfoil, with $T1 iteration of $PREC1L and $T2 iteration of $PREC2L precision"
else
  echo "Running only $PREC1L precision Airfoil, with $T1 iteration"
fi

cd ../$PREC1
echo "Running in $PREC1L precision"
mpirun -np 64 --oversubscribe ~/numawrap ./airfoil_mpi_genseq -i $T1 -o p_q_2.8m-${PREC1L}_it${T1}.h5 > $O1

if [[ "$PREC2" != "no" ]]; 
then
  cd ../$PREC2
  echo "Running in $PREC2L precision"
  mpirun -np 64 --oversubscribe ~/numawrap ./airfoil_mpi_genseq -i $T2 -d ../${PREC1}/p_q_2.8m-${PREC1L}_it${T1}.h5 -o p_q_2.8m-${T1}${PREC1L}_${T2}${PREC2L}.h5 > $O2
fi

cd ../compr
echo "Comparing results"
if [[ "$PREC2" != "no" ]]; 
then
  mpirun -np 1 --oversubscribe ~/numawrap ./airfoil_mpi_genseq -A ../${PREC2}/p_q_2.8m-${T1}${PREC1L}_${T2}${PREC2L}.h5 -B ${P_q_REF} -o airfoil_reldiff_p_q_2.8m-${T1}${PREC1}_${T2}${PREC2}.vtk > $O3
else
  mpirun -np 1 --oversubscribe ~/numawrap ./airfoil_mpi_genseq -A ../${PREC1}/p_q_2.8m-${PREC1L}_it${T1}.h5 -B ${P_q_REF} -o airfoil_reldiff_p_q_2.8m-${T1}${PREC1}}.vtk > $O3
fi

MTR1=$(cat $O1 | sed -nr 's/Max total runtime = (.*)/\1/p')
MTR2=$(cat $O2 | sed -nr 's/Max total runtime = (.*)/\1/p')
SMTR=$(awk "BEGIN{ print ${MTR1} + ${MTR2} }")
echo "Sum max total runtime: ${SMTR}"
cat $O3 | tail -n 6 | head -n 4

if [[ "$PREC2" != "no" ]]; 
then
  echo "Relative difference mesh results in: $(pwd)/airfoil_reldiff_p_q_2.8m-${T1}${PREC1}_${T2}${PREC2}.vtk"
else
  echo "Relative difference mesh results in: $(pwd)/airfoil_reldiff_p_q_2.8m-${T1}${PREC1}.vtk"
fi
