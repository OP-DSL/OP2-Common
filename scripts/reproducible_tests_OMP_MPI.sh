#!/bin/bash

#reproducible bulid and test script for MPI genseq

NP=2 #number of mpi processes
NTH=32 # number of OMP threads
N=5 #number of tests

MG_CFD_INSTALL_PATH=/home/sikba/MG-CFD-app-OP2
#MG_CFD_APP_PATH=/home/sikba/Rotor37_1M
MG_CFD_APP_PATH=/home/sikba/Rotor37_8M

outputdir=${OP2_INSTALL_PATH}/../scripts/reprotests/MPI-OMP/test-NP${NP}_NTH${NTH}_$(date "+%Y-%m-%d--%H_%M")
mkdir -p ${outputdir}

norepfile=${outputdir}/norepro.log
temparrayfile=${outputdir}/temparray.log
greedycolfile=${outputdir}/greedycol.log   
distfile=${outputdir}/distcol.log    
buildfile=${outputdir}/build.log    


echo "build log file: $buildfile"
echo "non reproducible file: $norepfile"
echo "temparray file: $temparrayfile"
echo "greedy coloring file: $greedycolfile "
echo "distributed coloring file: $distfile"

#generate nonreproducible files and compile them
echo "generateing nonreproducible apps"
cd ${OP2_INSTALL_PATH}/../translator/c/python
sed -i 's/reproducible = ./reproducible = 0/g' op2_gen_common.py
sed -i 's/repr_temp_array = ./repr_temp_array = 0/g' op2_gen_common.py
sed -i 's/repr_coloring = ./repr_coloring = 0/g' op2_gen_common.py

cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp
${OP2_INSTALL_PATH}/../translator/c/python/op2.py airfoil.cpp &>> $buildfile
make airfoil_mpi_openmp &>> $buildfile
mv airfoil_mpi_openmp airfoil_mpi_openmp_norepro

cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/
${OP2_INSTALL_PATH}/../translator/c/python/op2.py aero.cpp &>> $buildfile
make aero_mpi_openmp &>> $buildfile
mv aero_mpi_openmp aero_mpi_openmp_norepro

cd ${MG_CFD_INSTALL_PATH}
./translate2op2.sh &>> $buildfile
make mpi_openmp &>> $buildfile
mv bin/mgcfd_mpi_openmp bin/mgcfd_mpi_openmp_norepro


#generate temparray files and compile them
echo "generateing temparray apps"
cd ${OP2_INSTALL_PATH}/../translator/c/python
sed -i 's/reproducible = ./reproducible = 1/g' op2_gen_common.py
sed -i 's/repr_temp_array = ./repr_temp_array = 1/g' op2_gen_common.py
sed -i 's/repr_coloring = ./repr_coloring = 0/g' op2_gen_common.py

cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp
${OP2_INSTALL_PATH}/../translator/c/python/op2.py airfoil.cpp &>> $buildfile
make airfoil_mpi_openmp &>> $buildfile
mv airfoil_mpi_openmp airfoil_mpi_openmp_temparray

cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/
${OP2_INSTALL_PATH}/../translator/c/python/op2.py aero.cpp &>> $buildfile
make aero_mpi_openmp &>> $buildfile
mv aero_mpi_openmp aero_mpi_openmp_temparray

cd ${MG_CFD_INSTALL_PATH}
./translate2op2.sh &>> $buildfile
make mpi_openmp &>> $buildfile
mv bin/mgcfd_mpi_openmp bin/mgcfd_mpi_openmp_temparray


#generate repr coloring files and compile them
echo "generateing repr coloring apps"
cd ${OP2_INSTALL_PATH}/../translator/c/python
sed -i 's/reproducible = ./reproducible = 1/g' op2_gen_common.py
sed -i 's/repr_temp_array = ./repr_temp_array = 0/g' op2_gen_common.py
sed -i 's/repr_coloring = ./repr_coloring = 1/g' op2_gen_common.py

cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp
${OP2_INSTALL_PATH}/../translator/c/python/op2.py airfoil.cpp &>> $buildfile
make airfoil_mpi_openmp &>> $buildfile
mv airfoil_mpi_openmp airfoil_mpi_openmp_reprcolor

cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/
${OP2_INSTALL_PATH}/../translator/c/python/op2.py aero.cpp &>> $buildfile
make aero_mpi_openmp &>> $buildfile
mv aero_mpi_openmp aero_mpi_openmp_reprcolor

cd ${MG_CFD_INSTALL_PATH}
./translate2op2.sh &>> $buildfile
make mpi_openmp &>> $buildfile
mv bin/mgcfd_mpi_openmp bin/mgcfd_mpi_openmp_genseq_reprcolor

echo "done"


#start of measurements


#echo "--- Creating colors for greedy coloring test ---"
#
#cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp/
#rm -f *coloring.h5
#cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/
#rm -f *coloring.h5
#cd ${MG_CFD_APP_PATH}
#rm -f *coloring.h5
#
#cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp/
#echo "airfoil"
#OMP_NUM_THREADS=$NTH mpirun -np 1 ./airfoil_mpi_openmp_reprcolor -op_repro_greedy_coloring > /dev/null
#cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/
#echo "aero"
#OMP_NUM_THREADS=$NTH mpirun -np 1 ./aero_mpi_openmp_reprcolor -op_repro_greedy_coloring > /dev/null
#cd ${MG_CFD_APP_PATH}
#echo "mg-cfd"
#OMP_NUM_THREADS=$NTH mpirun -np 1 ../MG-CFD-app-OP2/bin/mgcfd_mpi_openmp_genseq_reprcolor -i input-mgcfd.dat -m ptscotch -op_repro_greedy_coloring > /dev/null



declare -a arr_airfoil_nonrepro
declare -a arr_airfoil_temparray
declare -a arr_airfoil_greedycol
declare -a arr_airfoil_distrcol

declare -a arr_aero_nonrepro
declare -a arr_aero_temparray
declare -a arr_aero_greedycol
declare -a arr_aero_distrcol

declare -a arr_mgcfd_nonrepro
declare -a arr_mgcfd_temparray
declare -a arr_mgcfd_greedycol
declare -a arr_mgcfd_distrcol


for i in $(eval echo "{1..$N}")
do
    echo "--- Repro measurements $i/$N ---"
    cd ${OP2_INSTALL_PATH}/../apps/c/airfoil/airfoil_reproducible/dp/
    echo -n "airfoil_mpi_openmp_norepro np${NP}, input: "
    ls -la | grep "new_grid.h5 " | grep -o "/new_grid.*" | tr '\n' '\0'
    echo ""
    echo "norepro"
    arr_airfoil_nonrepro+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./airfoil_mpi_openmp_norepro 2>&1 | tee -a $norepfile | grep -oP "Max total runtime = \K.*"))
    echo "temparray"
    arr_airfoil_temparray+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./airfoil_mpi_openmp_temparray 2>&1 | tee -a $temparrayfile | grep -oP "Max total runtime = \K.*"))
    echo "greedycol"
    arr_airfoil_greedycol+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./airfoil_mpi_openmp_reprcolor -op_repro_greedy_coloring 2>&1 | tee -a $greedycolfile | grep -oP "Max total runtime = \K.*"))
    echo "distcol"
    arr_airfoil_distrcol+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./airfoil_mpi_openmp_reprcolor 2>&1 | tee -a $distfile | grep -oP "Max total runtime = \K.*"))
    
    cd ${OP2_INSTALL_PATH}/../apps/c/aero/aero_reproducible/dp/

    echo -n "Aero_mpi_openmp np${NP}, input: "
    ls -la | grep "FE_grid.h5 " | grep -o "/FE_grid.*" | tr '\n' '\0'
    echo ""
    echo "norepro"
    arr_aero_nonrepro+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./aero_mpi_openmp_norepro 2>&1 | tee -a $norepfile | grep -oP "Max total runtime = \K.*"))
    echo "temparray"
    arr_aero_temparray+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./aero_mpi_openmp_temparray 2>&1 | tee -a $temparrayfile | grep -oP "Max total runtime = \K.*"))
    echo "greedycol"
    arr_aero_greedycol+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./aero_mpi_openmp_reprcolor -op_repro_greedy_coloring 2>&1 | tee -a $greedycolfile | grep -oP "Max total runtime = \K.*"))
    echo "distcol"
    arr_aero_distrcol+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2 ./aero_mpi_openmp_reprcolor 2>&1 | tee -a $distfile | grep -oP "Max total runtime = \K.*"))


    cd ${MG_CFD_APP_PATH}
    echo -n "mg-cfd_mpi np${NP}, input loc: ${MG_CFD_APP_PATH}"
    echo " "
    echo "norepro"
    arr_mgcfd_nonrepro+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2  ../MG-CFD-app-OP2/bin/mgcfd_mpi_openmp_norepro -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 2>&1 | tee -a $norepfile | grep -oP "Max total runtime = \K.*"))
    echo "temparray"
    arr_mgcfd_temparray+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2  ../MG-CFD-app-OP2/bin/mgcfd_mpi_openmp_temparray -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 2>&1 | tee -a $temparrayfile | grep -oP "Max total runtime = \K.*"))
    echo "greedycol"
    arr_mgcfd_greedycol+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2  ../MG-CFD-app-OP2/bin/mgcfd_mpi_openmp_genseq_reprcolor -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 -op_repro_greedy_coloring 2>&1 | tee -a $greedycolfile | grep -oP "Max total runtime = \K.*"))
    echo "distcol"
    arr_mgcfd_distrcol+=($(OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2  ../MG-CFD-app-OP2/bin/mgcfd_mpi_openmp_genseq_reprcolor -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 2>&1 | tee -a $distfile | grep -oP "Max total runtime = \K.*"))
    
#OMP_NUM_THREADS=$NTH mpirun -mca io ^ompio -np $NP ~/numawrap_omp2  ../MG-CFD-app-OP2/bin/mgcfd_mpi_omp_reprcolor -i input-mgcfd.dat -m ptscotch OP_PART_SIZE=2048 >> $distfile
done

#print results
echo -e "======= Max total runtimes ====== \n nonrepro ; temparray ; greedycol ; distrcol\n"

echo " Airfoil "
for index in "${!arr_airfoil_nonrepro[@]}"; 
do
    echo "${arr_airfoil_nonrepro[$index]};${arr_airfoil_temparray[$index]};${arr_airfoil_greedycol[$index]};${arr_airfoil_distrcol[$index]}"; 
done

echo -e "\n\n Aero "
for index in "${!arr_aero_nonrepro[@]}"; 
do
    echo "${arr_aero_nonrepro[$index]};${arr_aero_temparray[$index]};${arr_aero_greedycol[$index]};${arr_aero_distrcol[$index]}"; 
done

echo -e "\n\n Mg-cfd "
for index in "${!arr_mgcfd_nonrepro[@]}"; 
do
    echo "${arr_mgcfd_nonrepro[$index]};${arr_mgcfd_temparray[$index]};${arr_mgcfd_greedycol[$index]};${arr_mgcfd_distrcol[$index]}"; 
done

