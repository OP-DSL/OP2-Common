#!/bin/bash

#reproducible bulid and test script for MPI genseq

MAX_NP=64
N=1 #number of tests

MG_CFD_INSTALL_PATH=/home/sikba/MG-CFD-app-OP2
#MG_CFD_APP_PATH=/home/sikba/Rotor37_1M
MG_CFD_APP_PATH=/home/sikba/Rotor37_8M

#generate temparray files and compile them
echo "generateing temparray apps"
cd ${OP2_INSTALL_PATH}/../translator/c/python
sed -i 's/reproducible = ./reproducible = 1/g' op2_gen_common.py
sed -i 's/repr_temp_array = ./repr_temp_array = 1/g' op2_gen_common.py
sed -i 's/repr_coloring = ./repr_coloring = 0/g' op2_gen_common.py

cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp
${OP2_INSTALL_PATH}/../translator/c/python/op2.py airfoil.cpp
make airfoil_mpi_genseq 
mv airfoil_mpi_genseq airfoil_mpi_genseq_temparray

cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/
${OP2_INSTALL_PATH}/../translator/c/python/op2.py aero.cpp
make aero_mpi_genseq
mv aero_mpi_genseq aero_mpi_genseq_temparray

cd ${MG_CFD_INSTALL_PATH}
./translate2op2.sh
make mpi
mv bin/mgcfd_mpi bin/mgcfd_mpi_temparray


#generate repr coloring files and compile them
echo "generateing repr coloring apps"
cd ${OP2_INSTALL_PATH}/../translator/c/python
sed -i 's/reproducible = ./reproducible = 1/g' op2_gen_common.py
sed -i 's/repr_temp_array = ./repr_temp_array = 0/g' op2_gen_common.py
sed -i 's/repr_coloring = ./repr_coloring = 1/g' op2_gen_common.py

cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp
${OP2_INSTALL_PATH}/../translator/c/python/op2.py airfoil.cpp 
make airfoil_mpi_genseq 
mv airfoil_mpi_genseq airfoil_mpi_genseq_reprcolor

cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/
${OP2_INSTALL_PATH}/../translator/c/python/op2.py aero.cpp 
make aero_mpi_genseq 
mv aero_mpi_genseq aero_mpi_genseq_reprcolor

cd ${MG_CFD_INSTALL_PATH}
./translate2op2.sh 
make mpi 
mv bin/mgcfd_mpi bin/mgcfd_mpi_genseq_reprcolor

echo "done"


declare -a arr_airfoil_temparray
declare -a arr_airfoil_greedycol
declare -a arr_airfoil_distrcol

declare -a arr_aero_temparray
declare -a arr_aero_greedycol
declare -a arr_aero_distrcol

declare -a arr_mgcfd_temparray
declare -a arr_mgcfd_greedycol
declare -a arr_mgcfd_distrcol

i=1
for i in $(eval echo "{1..$N}")
do
    for ((np = 1; np <= $MAX_NP; np = np * 2))
    do

        echo "--- Repro measurements $np/$MAX_NP test $i/$N ---"
        cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp/
        echo -n "Airfoil_mpi_genseq np${np}, input: "
        ls -la | grep "new_grid.h5 " | grep -o "/new_grid.*" | tr '\n' '\0'
        echo ""
        echo "temparray"
        arr_airfoil_temparray+=($(mpirun -mca io ^ompio -np $np ~/numawrap ./airfoil_mpi_genseq_temparray | grep -oP "of only coloring = \K.*"))
        if [[ $np == 1 ]]
        then
            echo "greedycol"
            arr_airfoil_greedycol+=($(mpirun -mca io ^ompio -np $np ~/numawrap ./airfoil_mpi_genseq_reprcolor -op_repro_greedy_coloring | grep -oP "of only coloring = \K.*"))
        fi
        echo "distcol"
        arr_airfoil_distrcol+=($(mpirun -mca io ^ompio -np $np ~/numawrap ./airfoil_mpi_genseq_reprcolor | grep -oP "of only coloring = \K.*"))
        
        cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/

        echo -n "Aero_mpi_genseq np${NP}, input: "
        ls -la | grep "FE_grid.h5 " | grep -o "/FE_grid.*" | tr '\n' '\0'
        echo ""
        
        echo "temparray"
        arr_aero_temparray+=($(mpirun -mca io ^ompio -np $np ~/numawrap ./aero_mpi_genseq_temparray | grep -oP "of only coloring = \K.*"))
        if [[ $np == 1 ]]
        then
            echo "greedycol"
            arr_aero_greedycol+=($(mpirun -mca io ^ompio -np $np ~/numawrap ./aero_mpi_genseq_reprcolor -op_repro_greedy_coloring | grep -oP "of only coloring = \K.*"))
        fi
        echo "distcol"
        arr_aero_distrcol+=($(mpirun -mca io ^ompio -np $np ~/numawrap ./aero_mpi_genseq_reprcolor | grep -oP "of only coloring = \K.*"))


        cd ${MG_CFD_APP_PATH}
        echo -n "mg-cfd_mpi np${NP}, input loc: ${MG_CFD_APP_PATH}"
        echo " "


        echo "temparray"
        arr_mgcfd_temparray+=($(mpirun -mca io ^ompio -np $np ~/numawrap ../MG-CFD-app-OP2/bin/mgcfd_mpi_temparray -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 | grep -oP "of only coloring = \K.*"))
        if [[ $np == 1 ]]
        then
            echo "greedycol"
            arr_mgcfd_greedycol+=($(mpirun -mca io ^ompio -np $np ~/numawrap ../MG-CFD-app-OP2/bin/mgcfd_mpi_genseq_reprcolor -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 | grep -oP "of only coloring = \K.*"))
        fi
        echo "distcol"
        arr_mgcfd_distrcol+=($(mpirun -mca io ^ompio -np $np ~/numawrap ../MG-CFD-app-OP2/bin/mgcfd_mpi_genseq_reprcolor -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 | grep -oP "of only coloring = \K.*"))

    done    
done

#print results
echo -e "======= Max total runtimes ====== \n nonrepro ; temparray ; greedycol ; distrcol\n"


i=0
echo "Airfoil temparray"
for it in $(eval echo "{1..$N}")
do
    for ((np = 1; np <= $MAX_NP; np = np * 2))
    do
        echo -n "${arr_airfoil_temparray[$i]};"
        let "i=i+1"
    done
    echo ""
done 

i=0
echo "Airfoil greedycol"
for it in $(eval echo "{1..$N}")
do
   
        echo -n "${arr_airfoil_greedycol[$i]};"
        let "i=i+1"
    echo ""
   
done 


i=0
echo "Airfoil distcol"
for it in $(eval echo "{1..$N}")
do
    for ((np = 1; np <= $MAX_NP; np = np * 2))
    do
        echo -n "${arr_airfoil_distrcol[$i]};"
        let "i=i+1"
    done
    echo ""
done 




i=0
echo "Aero temparray"
for it in $(eval echo "{1..$N}")
do
    for ((np = 1; np <= $MAX_NP; np = np * 2))
    do
        echo -n "${arr_aero_temparray[$i]};"
        let "i=i+1"
    done
    echo ""
done 

i=0
echo "Aero greedycol"
for it in $(eval echo "{1..$N}")
do
   
        echo -n "${arr_aero_greedycol[$i]};"
        let "i=i+1"
    echo ""
   
done 


i=0
echo "Aero distcol"
for it in $(eval echo "{1..$N}")
do
    for ((np = 1; np <= $MAX_NP; np = np * 2))
    do
        echo -n "${arr_aero_distrcol[$i]};"
        let "i=i+1"
    done
    echo ""
done 




i=0
echo "Mg-cfd temparray"
for it in $(eval echo "{1..$N}")
do
    for ((np = 1; np <= $MAX_NP; np = np * 2))
    do
        echo -n "${arr_mgcfd_temparray[$i]};"
        let "i=i+1"
    done
    echo ""
done 

i=0
echo "Mg-cfd greedycol"
for it in $(eval echo "{1..$N}")
do
   
        echo -n "${arr_mgcfd_greedycol[$i]};"
        let "i=i+1"
    echo ""
   
done 


i=0
echo "Mg-cfd distcol"
for it in $(eval echo "{1..$N}")
do
    for ((np = 1; np <= $MAX_NP; np = np * 2))
    do
        echo -n "${arr_mgcfd_distrcol[$i]};"
        let "i=i+1"
    done
    echo ""
done 

