/* -------------------- Helper Functions -------------------- */

def getBinary(mesh, optimisation) {
    if (mesh in ["plain"] && optimisation.startsWith("mpi")) {
        return "aero_par_${optimisation}"
    }
    return "aero_${optimisation}"
}

def getDir(mesh) {
    return "OP2-Common/apps/c/aero/aero_${mesh}"
}

def runAero(mesh, optimisation) {
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

def validateAero(mesh, optimisation, grepWord="PASSED") {
    def binary = getBinary(mesh, optimisation)
    def dirPath = getDir(mesh)
    
    sh """
    grep -q '${grepWord}' ${dirPath}/output_${mesh}_${binary}.txt
    """
}

def archiveAero(mesh, optimisation, artifactDir) {
    def binary = getBinary(mesh, optimisation)
    def dirPath = getDir(mesh)
    
    sh """
    cp ${dirPath}/output_${mesh}_${binary}.txt ${artifactDir}/
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
                                if (pair.length == 2) env[pair[0]] = pair[1]
                            }
                        }
                    }
                }
                stage('Create Artifact Folder') {
                    steps {
                        script {
                            env.RUN_TIMESTAMP = new Date().format("yyyyMMdd-HHmmss")
                            env.ARTIFACT_BASE = "artifacts/aero-${env.RUN_TIMESTAMP}"
                            sh "mkdir -p ${env.ARTIFACT_BASE}"
                            echo "Artifacts will be saved in: ${env.ARTIFACT_BASE}"
                        }
                    }
                }
                stage('Checkout OP2-Common') {
                    steps {
                        dir('OP2-Common') {
                            git url: 'https://github.com/OP-DSL/OP2-Common.git', branch: 'OP2_refactor'
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

        stage('Build aero C') {
            parallel {
                stage("Build aero_plain") {
                    steps {
                        script {
                            def targets = [
                                "aero_seq", "aero_genseq", "aero_openmp", "aero_cuda",
                                "aero_par_mpi_seq", "aero_par_mpi_genseq", "aero_par_mpi_openmp", "aero_par_mpi_cuda"
                            ]
                            dir("OP2-Common/apps/c/aero/aero_plain") {
                                sh "make clean"
                                for (t in targets) sh "make ${t}"
                            }
                        }
                    }
                }
                stage("Build aero_hdf5") {
                    steps {
                        script {
                            def targets = [
                                "aero_seq", "aero_genseq", "aero_openmp", "aero_cuda",
                                "aero_mpi_seq", "aero_mpi_genseq", "aero_mpi_openmp", "aero_mpi_cuda"
                            ]
                            dir("OP2-Common/apps/c/aero/aero_hdf5") {
                                sh "make clean"
                                for (t in targets) sh "make ${t}"
                            }
                        }
                    }
                }
            }
        }

        stage('Download Meshes') {
            parallel {
                stage('Download plain mesh') {
                    steps {
                        dir("OP2-Common/apps/c/aero/aero_plain") {
                            sh 'rm -f FE_grid.dat && wget -nc https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/aero/FE_grid.dat'
                        }
                    }
                }
                stage('Download hdf5 mesh') {
                    steps {
                        dir("OP2-Common/apps/c/aero/aero_hdf5") {
                            sh 'rm -f FE_grid.h5 && wget -nc https://github.com/ZamanLantra/OP2-ci_meshes/raw/refs/heads/main/aero/FE_grid.h5'
                        }
                    }
                }
            }
        }

        stage('Aero Sequential Tests') {
            matrix {
                axes {
                    axis { name 'MESH'; values 'plain', 'hdf5' }
                    axis { name 'OPTIMISATION'; values 'seq', 'genseq' }
                }
                stages {
                    stage('Run Aero') {
                        steps { script { runAero(MESH, OPTIMISATION) } }
                    }
                    stage('Validate Output') {
                        steps { script { validateAero(MESH, OPTIMISATION) } }
                    }
                    stage('Archive Output') {
                        steps { script { archiveAero(MESH, OPTIMISATION, env.ARTIFACT_BASE) } }
                    }
                }
            }
        }

        stage('Aero Parallel Tests') {
            matrix {
                axes {
                    axis { name 'MESH'; values 'plain', 'hdf5' }
                    axis {
                        name 'OPTIMISATION'
                        values 'openmp', 'cuda', 'mpi_seq', 'mpi_genseq', 'mpi_openmp', 'mpi_cuda'
                    }
                }
                stages {
                    stage('Run Aero') {
                        steps { script { runAero(MESH, OPTIMISATION) } }
                    }
                    stage('Validate Output') {
                        steps { script { validateAero(MESH, OPTIMISATION) } }
                    }
                    stage('Archive Output') {
                        steps { script { archiveAero(MESH, OPTIMISATION, env.ARTIFACT_BASE) } }
                    }
                }
            }
        }
    }
}