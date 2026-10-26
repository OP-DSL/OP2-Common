/* MG-CFD (M6 wing and Rotor37 1M) through CTest: OP2 is configured with its
   apps and their tests (tests/apps), every variant built, and each run to
   its own validation against a downloaded mesh. */

pipeline {
    agent any

    // The defaults are the CI machine's; a test instance points them at its
    // own clone and environment.
    parameters {
        string(name: 'OP2_REPO', defaultValue: 'https://github.com/OP-DSL/OP2-Common.git',
               description: 'OP2 repository to build and test')
        string(name: 'OP2_BRANCH', defaultValue: 'OP2_refactor', description: 'Its branch')
        string(name: 'MGCFD_REPO', defaultValue: 'https://github.com/warwick-hpsc/MG-CFD-app-OP2.git',
               description: 'MG-CFD repository whose C++ app is tested')
        string(name: 'MGCFD_BRANCH', defaultValue: 'OP2_refactor', description: 'Its branch')
        string(name: 'ENV_SCRIPT', defaultValue: '/home/zl/work3/OP2-Common/scripts/zam_gnu',
               description: 'Script sourced for the compilers and dependency prefixes')
    }

    stages {
        stage('Setup') {
            parallel {
                stage('Create Artifact Folder') {
                    steps {
                        script {
                            env.RUN_TIMESTAMP = new Date().format("yyyyMMdd-HHmmss")
                            env.ARTIFACT_BASE = "artifacts/mgcfd-${env.RUN_TIMESTAMP}"

                            sh "mkdir -p ${env.ARTIFACT_BASE}"
                            echo "Artifacts will be saved in: ${env.ARTIFACT_BASE}"
                        }
                    }
                }
                stage('Checkout MG-CFD repo') {
                    steps {
                        dir('MG-CFD-app-OP2') {
                            git url: params.MGCFD_REPO, branch: params.MGCFD_BRANCH
                        }
                    }
                }
                stage('Checkout OP2-Common') {
                    steps {
                        dir('OP2-Common') {
                            git url: params.OP2_REPO, branch: params.OP2_BRANCH
                        }
                    }
                }
            }
        }

        // Each shell step sources ENV_SCRIPT for the compilers and dependency
        // prefixes.
        stage('Configure OP2') {
            steps {
                dir('OP2-Common') {
                    sh '''
                    . /etc/profile
                    . "$ENV_SCRIPT"
                    rm -rf build
                    cmake -B build -DCMAKE_BUILD_TYPE=Release \\
                        -DPython3_EXECUTABLE="$(command -v python3)" \\
                        -DCMAKE_CUDA_ARCHITECTURES=native \\
                        -DHDF5_ROOT="$HDF5_PAR_INSTALL_PATH" \\
                        -DParMETIS_ROOT="$PARMETIS_INSTALL_PATH" \\
                        -DMETIS_ROOT="${METIS_INSTALL_PATH:-$PARMETIS_INSTALL_PATH}" \\
                        -DKaHIP_ROOT="$KAHIP_INSTALL_PATH" \\
                        -DOP2_ENABLE_APPS=ON -DOP2_ENABLE_TESTS=ON -DOP2_ENABLE_APP_TESTS=ON \\
                        -DOP2_TEST_MGCFD_DIR="$WORKSPACE/MG-CFD-app-OP2"
                    '''
                }
            }
        }

        stage('Build OP2 and the apps') {
            steps {
                dir('OP2-Common') {
                    sh '. /etc/profile; . "$ENV_SCRIPT"; cmake --build build -j'
                }
            }
        }

        // A failed test marks the build unstable through the JUnit report.
        stage('Run mgcfd') {
            steps {
                dir('OP2-Common') {
                    sh ". /etc/profile; . \"\$ENV_SCRIPT\"; ctest --test-dir build -L apps -R '^mgcfd_' -j8 --output-on-failure --output-junit mgcfd.xml || true"
                }
            }
            post {
                always {
                    junit testResults: 'OP2-Common/build/mgcfd.xml', allowEmptyResults: false
                    sh "cp OP2-Common/build/Testing/Temporary/LastTest*.log ${env.ARTIFACT_BASE}/ || true"
                    archiveArtifacts artifacts: "${env.ARTIFACT_BASE}/*", allowEmptyArchive: true
                }
            }
        }
    }
}
