# Jenkins
https://www.jenkins.io/doc/book/installing/linux/

# Run jenkins
java -jar ~/jenkins/jenkins.war --httpPort=8102

# Port forward, if running remotely
ssh -N -L 8102:localhost:8102 <user>@<machine>
# Test a pipeline locally
The pipelines take `OP2_REPO`, `OP2_BRANCH` and `ENV_SCRIPT` (and, for MG-CFD,
`MGCFD_REPO` and `MGCFD_BRANCH`) as build parameters, defaulting to the CI
machine's, so a throwaway Jenkins can run them against a local clone:

```
mkdir -p ~/jenkins-local/home/plugins && cd ~/jenkins-local
curl -sSLO https://get.jenkins.io/war-stable/latest/jenkins.war
curl -sSL -o plugin-manager.jar https://github.com/jenkinsci/plugin-installation-manager-tool/releases/download/2.15.0/jenkins-plugin-manager-2.15.0.jar
java -jar plugin-manager.jar --war jenkins.war --plugin-download-directory home/plugins \
    --plugins workflow-aggregator git junit pipeline-stage-view
JENKINS_HOME=$PWD/home java -Djenkins.install.runSetupWizard=false \
    -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true \
    -jar jenkins.war --httpPort=8102 --httpListenAddress=127.0.0.1
```

Then add a Pipeline job whose definition is "Pipeline script from SCM": Git,
repository `file:///path/to/OP2-Common`, the branch to test, script path
`tests/jenkins/<app>.groovy`.  Its first build registers the parameters (and
runs with the defaults); build it again with parameters `OP2_REPO` set to the
same `file://` URL, `OP2_BRANCH` to the branch, and `ENV_SCRIPT` to a script
exporting `PATH` (a Python with the translator's requirements first),
`HDF5_PAR_INSTALL_PATH`, `PARMETIS_INSTALL_PATH`, `METIS_INSTALL_PATH` and
`KAHIP_INSTALL_PATH` as needed.  A machine without a GPU runs the CPU
variants only.
