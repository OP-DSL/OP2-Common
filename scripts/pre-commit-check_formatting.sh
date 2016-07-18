#!/usr/bin/env python

import subprocess
retcode = subprocess.call("which git-clang-format > /dev/null", shell=True)
if retcode == 0:
  output = subprocess.check_output(["git", "clang-format", "--diff"])
  print output
  if output not in ['no modified files to format\n', 'clang-format did not modify any files\n']:
    print "Need changes to adhere to OP2 coding guidelines."
    print "Run git clang-format -f, check if the formatting is acceptable, then commit.\n"
    exit(1)
  else:
    exit(0)
else:
  print 'Cannot find git-clang-format in PATH'
  print 'Install and add git-clang-format to PATH to format code changes to conform to code formatting guidelines'
