#!/bin/bash
#Uses clang-format to format code to conform to the OP2 coding guidelines
# ... currently only applys to files within the current directory
for file in ./*.cu ./*.cpp ./*.h ./*.hpp; do clang-format "$file" > "$file"_temp; mv "$file"_temp "$file"; done
