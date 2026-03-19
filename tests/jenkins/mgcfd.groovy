/* -------------------- Helper Functions -------------------- */

def getBinary(optimisation) {
    return "euler3d_${optimisation}"
}

def runMgcfd(dataset, optimisation) {

    def binary = getBinary(optimisation)

    dir(dataset) {

        sh """
        echo "Running ${binary} on ${dataset}"
        """

        if (binary.contains("mpi")) {
            sh "mpirun -np 8 ../MG-CFD-app-OP2/${binary} -i input.dat -v > output_${dataset}_${binary}.txt 2>&1"
        } else {
            sh "../MG-CFD-app-OP2/${binary} -i input.dat -v > output_${dataset}_${binary}.txt"
        }
    }
}

def validateMgcfd(dataset, optimisation) {

    def binary = getBinary(optimisation)

    sh """
    grep -q "Validation passed" ${dataset}/output_${dataset}_${binary}.txt
    """
}

def archiveMgcfd(dataset, optimisation, artifactDir) {

    def binary = getBinary(optimisation)

    sh """
    cp ${dataset}/output_${dataset}_${binary}.txt ${artifactDir}/
    """

    archiveArtifacts artifacts: "${artifactDir}/output_${dataset}_${binary}.txt", allowEmptyArchive: true
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
                            env.ARTIFACT_BASE = "artifacts/mgcfd-${env.RUN_TIMESTAMP}"

                            sh "mkdir -p ${env.ARTIFACT_BASE}"
                            echo "Artifacts stored in ${env.ARTIFACT_BASE}"
                        }
                    }
                }
                stage('Checkout MG-CFD repo') {

                    steps {
                        dir('MG-CFD-app-OP2') {
                            git url: 'https://github.com/warwick-hpsc/MG-CFD-app-OP2.git', branch: 'OP2_refactor'
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
                    sh 'unset OP_AUTO_SOA'
                    sh 'make config'
                    sh 'make clean'
                    sh 'make config'
                    sh 'make -j'
                }
            }
        }

        stage('Build mgcfd') {
            steps {
                script {
                    def targets = [
                        "euler3d_seq",
                        "euler3d_genseq",
                        "euler3d_openmp",
                        "euler3d_cuda",
                        "euler3d_mpi_seq",
                        "euler3d_mpi_genseq",
                        "euler3d_mpi_openmp",
                        "euler3d_mpi_cuda"
                    ]

                    dir("MG-CFD-app-OP2") {
                        sh "make clean"

                        for (t in targets) {
                            sh "make ${t}"
                        }
                    }
                }
            }
        }

        stage('Download meshes') {
            parallel {
                stage('Download M6_wing') {
                    steps {
                        sh '''
                        if [ ! -d M6_wing ]; then
                            wget https://warwick.ac.uk/fac/sci/dcs/research/systems/hpsc/software/m6_wing.tar.gz
                            tar -xvf m6_wing.tar.gz
                            rm m6_wing.tar.gz
                        fi
                        '''
                    }
                }
                stage('Download Rotor37_1M') {
                    steps {
                        sh '''
                        if [ ! -d Rotor37_1M ]; then
                            wget https://warwick.ac.uk/fac/sci/dcs/research/systems/hpsc/software/rotor37_1m.tar.gz
                            tar -xvf rotor37_1m.tar.gz
                            rm rotor37_1m.tar.gz
                        fi
                        '''
                    }
                }
            }
        }

        stage('MGCFD Sequential Tests') {
            matrix {
                axes {
                    axis {
                        name 'DATASET'
                        values 'M6_wing', 'Rotor37_1M'
                    }
                    axis {
                        name 'OPTIMISATION'
                        values 'seq', 'genseq'
                    }
                }

                stages {
                    stage('Run mgcfd') {
                        steps {
                            script {
                                runMgcfd(DATASET, OPTIMISATION)
                            }
                        }
                    }

                    stage('Validate output') {
                        steps {
                            script {
                                validateMgcfd(DATASET, OPTIMISATION)
                            }
                        }
                    }

                    stage('Archive output') {
                        steps {
                            script {
                                archiveMgcfd(DATASET, OPTIMISATION, env.ARTIFACT_BASE)
                            }
                        }
                    }
                }
            }
        }

        stage('MGCFD Parallel Tests') {
            matrix {
                axes {
                    axis {
                        name 'DATASET'
                        values 'M6_wing', 'Rotor37_1M'
                    }
                    axis {
                        name 'OPTIMISATION'
                        values 'openmp', 'cuda', 'mpi_seq', 'mpi_genseq', 'mpi_openmp', 'mpi_cuda'
                    }
                }

                stages {
                    stage('Run mgcfd') {
                        steps {
                            script {
                                runMgcfd(DATASET, OPTIMISATION)
                            }
                        }
                    }

                    stage('Validate output') {
                        steps {
                            script {
                                validateMgcfd(DATASET, OPTIMISATION)
                            }
                        }
                    }

                    stage('Archive output') {
                        steps {
                            script {
                                archiveMgcfd(DATASET, OPTIMISATION, env.ARTIFACT_BASE)
                            }
                        }
                    }
                }
            }
        }
    }
}