### Scripts

This directory contains various scripts used for testing the build and correct
execution of OP2 and its example applications. These are provided and commited
to the repository simply as a referance to any one who is installing and using OP2.


#### Files

source_intel, source_pgi -- example files that can be sourced to set the required variables to compile and install with
Intel and PGI compilers respectively.

ruby.sh -- builds OP2 library using cmake, then builds OP2 example applications using cmake for
ruby.oerc.ox.ac.uk
octon.sh -- builds OP2 library using cmake, then builds OP2 example applications using cmake for octon.arc.ox.ac.uk

test_cmake.sh -- builds C libs and apps with cmake and tests them (note this does not include Fortran libs and apps)
test_makefiles.sh -- builds and tests OP2 lib and apps with plain Makefiles (including Fortran libs and apps)
