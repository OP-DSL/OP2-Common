# Jenkins
https://www.jenkins.io/doc/book/installing/linux/

# Run jenkins
java -jar ~/jenkins/jenkins.war --httpPort=8102

# Port forward, if running remotely
ssh -N -L 8102:localhost:8102 <user>@<machine>