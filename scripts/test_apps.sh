#!/bin/bash

#
# Test OP2 example applications
#

# ============= Setup these params before running =============

export SOURCE_FILE=source_intel_18

airfoil_mesh=~/prev/airfoil-mesh-datafiles/new_grid.dat
fe_mesh=~/prev/airfoil-mesh-datafiles/FE_grid.dat

airfoil_mesh_h5=~/prev/airfoil-mesh-datafiles/new_grid.h5
fe_mesh_h5=~/prev/airfoil-mesh-datafiles/FE_grid.h5

omp_num_threads=2
num_procs=2

# ==============================================================

export LIBCLANG_PATH=/usr/lib/llvm-14/lib/libclang-14.so.1
export EXTRA_CFLAGS=-std=c++17
export EXTRA_CXXFLAGS=-std=c++17

alias python=python3
export CURRENT_DIR=$PWD
cd ../op2
export OP2_INSTALL_PATH=$PWD
cd $OP2_INSTALL_PATH
cd ../apps
export OP2_APPS_DIR=$PWD

# Set the output file path
# Check this file for failures after run
output_file="$CURRENT_DIR/test_apps_output.txt"

# Function to validate command and append result to output file
function validate {
  echo "$1" >> "$output_file"
  $1 > perf_out
  if grep -q "PASSED" perf_out; then
    echo "PASSED" >> "$output_file"
    echo "PASSED"
  else
    echo "TEST FAILED" >> "$output_file"
    echo "TEST FAILED"
    # exit 1
  fi
  rm perf_out
}


echo "***********************> Building C back-end libs <*******************"
source $CURRENT_DIR/$SOURCE_FILE
cd $OP2_INSTALL_PATH

make clean
output=$(make config)
# Extract the line with "Buildable C library variants"
options_line=$(echo "$output" | grep -E "Buildable library variants" | awk '{gsub(/f_/,""); for(i=4;i<=NF;i++) printf "%s ", $i}')
cmd="make "${options_line%" "}""
echo $cmd
# $cmd


echo "***********************> Building apps <*******************"

apps=(aero airfoil jac jac reduction)
folders=(aero airfoil jac1 jac2 reduction)

has_hdf5=(1 1 0 0 0)
has_sp_dp=(0 1 1 0 0)

hdf5=(plain hdf5)
sp_dp=(sp dp)

target_dir=""
version="c"
app_path=""

for ((i = 0; i < ${#folders[@]}; i++)); do

  folder=${folders[i]}
  app=${apps[i]}

  for dat_type in "${hdf5[@]}"; do

    if [[ ${has_hdf5[i]} == 1 ]]; then
      mode="${folder}_${dat_type}"
    else
      mode=""
    fi 

      for prec in "${sp_dp[@]}"; do
        if [[ ${has_sp_dp[i]} == 1 ]]; then
          app_path="/${version}/${folder}/${mode}/${prec}"
        else
          prec=""
          app_path="/${version}/${folder}/${mode}/${prec}"
        fi

        target_dir="$OP2_APPS_DIR/$app_path"
        # Change directory to the target directory
        cd "$target_dir"

        input=$(make config)
        

        # Extract versions for app
        versions_app=$(echo "$input" | grep -E "Buildable app variants for $app:" | awk -F': ' '{print $2}' | sed -E "s/(\S+)/${app}_\1/g")

        # Extract versions for app_par
        versions_app_par=$(echo "$input" | grep -E "Buildable app variants for ${app}_par:" | awk -F': ' '{print $2}' | sed -E "s/_/_/g; s/(\S+)/${app}_par_\1/g")

        # Build app versions
        cmd="make clean $versions_app"
        echo $cmd
        $cmd

        cmd="make $versions_app_par"
        echo $cmd
        $cmd


        if [[ ${app} == "aero" || ${app} == "airfoil" || ${app} == "jac" ]]; then

          # cmd="rm -rf new_grid.*"
          # echo $cmd
          # $cmd

          if [[ ${dat_type} == "hdf5" ]]; then
            cmd="ln -s $airfoil_mesh new_grid.h5"
          else
            cmd="ln -s $airfoil_mesh new_grid.dat"
          fi

          echo $cmd
          $cmd
        fi

        if [[ ${app} == "aero" ]]; then

          # cmd="rm -rf FE_grid.*"
          # echo $cmd
          # $cmd

          if [[ ${dat_type} == "hdf5" ]]; then
            cmd="ln -s $airfoil_mesh FE_grid.h5"
          else
            cmd="ln -s $airfoil_mesh FE_grid.dat"
          fi
          
          echo $cmd
          $cmd
        fi

        echo ""  >> "$output_file"
        echo "***********************> Executing ${app} <*******************"  >> "$output_file"
        echo "Targer dir:$target_dir App:${app}" >> "$output_file"

        echo ""
        echo "***********************> Executing ${app} <*******************"
        echo "Targer dir:$target_dir App:${app}"

        cd $target_dir

        echo "${app}_seq"
        validate "./${app}_seq"

        echo "${app}_genseq"
        validate "./${app}_genseq"

        echo "${app}_openmp"
        export OMP_NUM_THREADS=${omp_num_threads}
        validate "./${app}_openmp"

        echo "${app}_cuda"
        validate "./${app}_cuda"

        echo "${app}_par_mpi_seq"
        validate "$MPI_INSTALL_PATH/bin/mpirun -np ${num_procs} ./${app}_par_mpi_seq"

        echo "${app}_par_mpi_genseq"
        validate "$MPI_INSTALL_PATH/bin/mpirun -np ${num_procs} ./${app}_par_mpi_genseq"

        echo "${app}_par_mpi_openmp"
        export OMP_NUM_THREADS=${omp_num_threads}
        validate "$MPI_INSTALL_PATH/bin/mpirun -np ${num_procs} ./${app}_par_mpi_openmp"

        echo "${app}_par_mpi_cuda"
        validate "$MPI_INSTALL_PATH/bin/mpirun -np ${num_procs} ./${app}_par_mpi_cuda"

        echo "***********************> ${app} execution done <*******************"  >> "$output_file"
        echo ""  >> "$output_file"

        echo "***********************> ${app} execution done <*******************"
        echo ""

        if [[ ${prec} == "" ]]; then
          break
        fi 
      done

      if [[ ${mode} == "" ]]; then
        break
      fi 
  done

done
