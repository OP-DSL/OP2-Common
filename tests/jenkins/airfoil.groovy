/* -------------------- Helper Functions -------------------- */

def getBinary(mesh, optimisation) {

    if (mesh in ["plain","tempdats"] && optimisation.startsWith("mpi")) {
        return "airfoil_par_${optimisation}"
    }

    return "airfoil_${optimisation}"
}

def getDir(mesh) {
    return "OP2-Common/apps/c/airfoil/airfoil_${mesh}/dp"
}

def runAirfoil(mesh, optimisation) {

    def binary = getBinary(mesh, optimisation)
    def dirPath = getDir(mesh)

    dir(dirPath) {

        if (binary.contains("mpi")) {
            sh "mpirun -np 8 ./${binary} > output_${mesh}_${binary}.txt"
        } else {
            sh "./${binary} > output_${mesh}_${binary}.txt"
        }
    }
}

def validateAirfoil(mesh, optimisation) {

    def binary = getBinary(mesh, optimisation)

    sh """
    grep -q 'This test is considered PASSED' \
    OP2-Common/apps/c/airfoil/airfoil_${mesh}/dp/output_${mesh}_${binary}.txt
    """
}

def archiveAirfoil(mesh, optimisation, artifactDir) {

    def binary = getBinary(mesh, optimisation)

    sh """
    cp OP2-Common/apps/c/airfoil/airfoil_${mesh}/dp/output_${mesh}_${binary}.txt ${artifactDir}/
    """

    archiveArtifacts artifacts: "${artifactDir}/output_${mesh}_${binary}.txt", allowEmptyArchive: true
}

/* -------------------- Pipeline -------------------- */
pipeline {
    agent any

    environment {
        OP2_COMPILER = "gnu"
        OMP_NUM_THREADS = "8"
    }

    stages {
        stage('Setup') {
            parallel {
                stage('Load OP2 Environment') {
                    steps {
                        script {
                            def vars = sh(
                                script: '''
                                #!/bin/bash
                                . /etc/profile
                                . /home/zl/work3/OP2-Common/scripts/zam_gnu
                                env
                                ''',
                                returnStdout: true
                            ).trim().split("\n")
                
                            for (v in vars) {
                                def pair = v.split("=",2)
                                if (pair.length == 2) {
                                    env[pair[0]] = pair[1]
                                }
                            }
                        }
                    }
                }
                stage('Create Artifact Folder') {
                    steps {
                        script {
                            env.RUN_TIMESTAMP = new Date().format("yyyyMMdd-HHmmss")
                            env.ARTIFACT_BASE = "artifacts/airfoil-${env.RUN_TIMESTAMP}"

                            sh "mkdir -p ${env.ARTIFACT_BASE}"
                            echo "Artifacts will be saved in: ${env.ARTIFACT_BASE}"
                        }
                    }
                }
                stage('Checkout OP2-Common') {
                    steps {
                        dir('OP2-Common') {
                            git url: 'https://github.com/OP-DSL/OP2-Common.git', branch: 'OP2_refactor'

                            sh '''
                            FILE=makefiles/dependencies/parmetis.mk
                
                            if ! grep -q "\\-lGKlib" "$FILE"; then
                                echo "Adding -lGKlib"
                
                                sed -i 's/^[[:space:]]*PARMETIS_LINK.*/  PARMETIS_LINK ?= -lparmetis -lmetis -lGKlib/' "$FILE"
                            fi
                
                            echo "Current line:"
                            grep PARMETIS_LINK "$FILE"
                            '''
                        }
                    }
                }
            }
        }

        stage('Build OP2') {
            steps {
                dir('OP2-Common/op2') {
                    sh 'make config'
                    sh 'make clean'
                    sh 'make config'
                    sh 'make -j'
                }
            }
        }
        
        stage('Build airfoil C') {
            parallel {
                stage("Build airfoil_plain") {
                    steps {
                        script {
                            def targets = [
                                "airfoil_seq",
                                "airfoil_genseq",
                                "airfoil_openmp",
                                "airfoil_cuda",
                                "airfoil_par_mpi_seq",
                                "airfoil_par_mpi_genseq",
                                "airfoil_par_mpi_openmp",
                                "airfoil_par_mpi_cuda"
                            ]

                            dir("OP2-Common/apps/c/airfoil/airfoil_plain/dp") {
                                sh "make clean"

                                for (t in targets) {
                                    sh "make ${t}"
                                }
                            }
                        }
                    }
                }
                stage("Build airfoil_hdf5") {
                    steps {
                        script {
                            def targets = [
                                "airfoil_seq",
                                "airfoil_genseq",
                                "airfoil_openmp",
                                "airfoil_cuda",
                                "airfoil_mpi_seq",
                                "airfoil_mpi_genseq",
                                "airfoil_mpi_openmp",
                                "airfoil_mpi_cuda"
                            ]

                            dir("OP2-Common/apps/c/airfoil/airfoil_hdf5/dp") {
                                sh "make clean"

                                for (t in targets) {
                                    sh "make ${t}"
                                }
                            }
                        }
                    }
                }
                stage("Build airfoil_tempdats") {
                    steps {
                        script {
                            def targets = [
                                "airfoil_seq",
                                "airfoil_genseq",
                                "airfoil_openmp",
                                "airfoil_cuda",
                                "airfoil_par_mpi_seq",
                                "airfoil_par_mpi_genseq",
                                "airfoil_par_mpi_openmp",
                                "airfoil_par_mpi_cuda"
                            ]

                            dir("OP2-Common/apps/c/airfoil/airfoil_tempdats/dp") {
                                sh "make clean"

                                for (t in targets) {
                                    sh "make ${t}"
                                }
                            }
                        }
                    }
                }
            }
        }
        stage('Download mesh') {
            parallel {
                stage('Download plain mesh') {
                    steps {
                        dir("OP2-Common/apps/c/airfoil/airfoil_plain/dp") {
                            sh 'rm -f new_grid.dat && wget -nc https://op-dsl.github.io/docs/OP2/new_grid.dat'
                        }
                    }
                }
                stage('Download hdf5 mesh') {
                    steps {
                        dir("OP2-Common/apps/c/airfoil/airfoil_hdf5/dp") {
                            sh 'rm -f new_grid.h5 && wget -nc -O new_grid.h5 https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/airfoil/new_grid_dp.h5'
                        }
                    }
                }
                stage('Download tempdats mesh') {
                    steps {
                        dir("OP2-Common/apps/c/airfoil/airfoil_tempdats/dp") {
                            sh 'rm -f new_grid.dat && wget -nc https://op-dsl.github.io/docs/OP2/new_grid.dat'
                        }
                    }
                }
            }
        }
        
        stage('Airfoil Sequential Tests') {
            matrix {
                axes {
                    axis {
                        name 'MESH'
                        values 'plain', 'hdf5', 'tempdats'
                    }
                    axis {
                        name 'OPTIMISATION'
                        values 'seq', 'genseq'
                    }
                }

                stages {
                    stage("Run airfoil") {
                        steps {
                            script {
                                runAirfoil(MESH, OPTIMISATION)
                            }
                        }
                    }

                    stage('Validate output') {
                        steps {
                            script {
                                validateAirfoil(MESH, OPTIMISATION)
                            }
                        }
                    }

                    stage('Archive output') {
                        steps {
                            script {
                                archiveAirfoil(MESH, OPTIMISATION, env.ARTIFACT_BASE)
                            }
                        }
                    }
                }
            }
        }

        stage('Airfoil Parallel Tests') {
            matrix {
                axes {
                    axis {
                        name 'MESH'
                        values 'plain', 'hdf5', 'tempdats'
                    }
                    axis {
                        name 'OPTIMISATION'
                        values 'openmp', 'cuda', 'mpi_seq', 'mpi_genseq', 'mpi_openmp', 'mpi_cuda'
                    }
                }

                stages {
                    stage("Run airfoil") {
                        steps {
                            script {
                                runAirfoil(MESH, OPTIMISATION)
                            }
                        }
                    }

                    stage('Validate output') {
                        steps {
                            script {
                                validateAirfoil(MESH, OPTIMISATION)
                            }
                        }
                    }

                    stage('Archive output') {
                        steps {
                            script {
                                archiveAirfoil(MESH, OPTIMISATION, env.ARTIFACT_BASE)
                            }
                        }
                    }
                }
            }
        }
    }
}

