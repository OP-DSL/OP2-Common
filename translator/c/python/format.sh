#!/bin/bash
#Uses clang-format to format code to conform to the OP2 coding guidelines
# ... currently only applys to files within the current directory
#for file in ./*.cu ./*.cpp ./*.h ./*.hpp; do clang-format "$file" > "$file"_temp; mv "$file"_temp "$file"; done

ls ./*.cu 2> /dev/null
if [ $? -eq 0 ]
then
  for file in ./*.cu; do clang-format -i "$file"; done
fi
ls ./*.c 2> /dev/null
if [ $? -eq 0 ]
then
  for file in ./*.c ; do clang-format -i "$file"; done
fi
ls ./*.cpp 2> /dev/null
if [ $? -eq 0 ]
then
  for file in ./*.cpp ; do clang-format -i "$file"; done
fi
ls ./*.h 2> /dev/null
if [ $? -eq 0 ]
then
  for file in ./*.h ; do clang-format -i "$file"; done
fi
ls ./*.hpp 2> /dev/null
if [ $? -eq 0 ]
then
  for file in ./*.hpp ; do clang-format -i "$file"; done
fi
