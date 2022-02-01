import re

from store import Program
from util import SourceBuffer


# Augment source program to use generated kernel hosts
def translateProgram(source: str, program: Program, soa: bool = False) -> str:
    buffer = SourceBuffer(source)

    # 1. Update const calls
    for const in program.consts:
        buffer.apply(
            const.loc.line - 1,
            lambda line: re.sub(r"op_decl_const\s*\(", f'op_decl_const2("{const.debug}", ', line),
        )

    # 2. Update loop calls
    for loop in program.loops:
        before, after = buffer.get(loop.loc.line - 1).split("op_par_loop", 1)
        after = re.sub(
            f"{loop.kernel}\s*,\s*", "", after, count=1
        )  # TODO: This assumes that the kernel arg is on the same line as the call
        buffer.update(loop.loc.line - 1, before + f"op_par_loop_{loop.name}" + after)

    # 3. Update headers
    index = buffer.search(r'\s*#include\s+"op_seq\.h"') + 2
    buffer.insert(index, '#ifdef OPENACC\n#ifdef __cplusplus\nextern "C" {\n#endif\n#endif\n')
    for loop in program.loops:
        prototype = f'void op_par_loop_{loop.name}(char const *, op_set{", op_arg" * len(loop.args)});\n'
        buffer.insert(index, prototype)
    buffer.insert(index, "#ifdef OPENACC\n#ifdef __cplusplus\n}\n#endif\n#endif\n")

    # 4. Update init call TODO: Use a line number from the program
    source = buffer.translate()
    if soa:
        source = re.sub(r"\bop_init\b\s*\((.*)\)", "op_init_soa(\\1,1)", source)
        source = re.sub(r"\bop_mpi_init\b\s*\((.*)\)", "op_mpi_init_soa(\\1,1)", source)

    # Substitude the op_seq.h include for op_lib_cpp.h
    source = re.sub(r'#include\s+"op_seq\.h"', '#include "op_lib_cpp.h"', source)

    return source
