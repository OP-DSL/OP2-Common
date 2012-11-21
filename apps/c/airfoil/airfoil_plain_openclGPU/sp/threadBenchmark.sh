#!/bin/bash

echo -n "" > thread.log ;
for i in $(seq 1 1 24); 
do
  export OMP_NUM_THREADS=$i ;
  for j in {1..4}; 
  do
    a=$(./airfoil_openmp | tail -n 1) ;
    echo -n "$a " >> thread.log ;
    echo $a
  done;
  echo -ne "\n" >> thread.log ;
done

