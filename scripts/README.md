### Scripts

This directory contains various scripts used for testing the build and correct 
execution of OP2 and its example applications. These are provided and commited 
to the repository simply as a referance to any one who is installing and using OP2. 
Currently all these scripts are specific to ruby.oerc.ox.ac.uk 

#### Files

source_intel, source_pgi -- example files that can be sourced to set the required variables to compile and install with 
Intel and PGI compilers respectively. 

ruby.sh -- builds OP2 library using cmake

ruby_apps.sh -- builds OP2 example applications using cmake 

test_script.sh -- tests OP2 apps built with cmake (does not include Fortran libs and apps)

test.sh -- builds and tests OP2 lib and apps with plain Makefiles (including Fortran libs and apps)
