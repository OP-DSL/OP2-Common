import re

from store import Program
from util import SourceBuffer


# Augment source program to use generated kernel hosts
def translateProgram(source: str, program: Program, force_soa: bool) -> str:
    buffer = SourceBuffer(source)

    # 1. Update const calls, right to left so a line's earlier columns stay put
    for const in sorted(program.consts, key=lambda const: (const.loc.line, const.loc.column), reverse=True):
        column = const.loc.column - 1
        buffer.apply(
            const.loc.line - 1,
            lambda line: line[:column]
            + re.sub(r"op_decl_const\s*\(", f'op_decl_const2("{const.ptr}", ', line[column:], count=1),
        )

    # 2. Update loop calls; the kernel, the first argument, may be on a later line
    for loop in program.loops:
        index = loop.loc.line - 1
        kernel = re.compile(rf"\b{re.escape(loop.kernel)}\s*,\s*")
        before, after = buffer.get(index).split("op_par_loop", 1)
        after, found = kernel.subn("", after, count=1)
        buffer.update(index, before + f"op_par_loop_{loop.name}" + after)
        while not found:
            index += 1
            line, found = kernel.subn("", buffer.get(index), count=1)
            buffer.update(index, line)

    # 3. Update headers
    index = buffer.search(r'\s*#include\s+"op_seq\.h"')
    assert index is not None
    index += 2

    buffer.insert(index, '#ifdef OPENACC\n#ifdef __cplusplus\nextern "C" {\n#endif\n#endif\n')
    for loop in program.loops:
        prototype = f'void op_par_loop_{loop.name}(char const *, op_set{", op_arg" * len(loop.args)});\n'
        buffer.insert(index, prototype)
    buffer.insert(index, "#ifdef OPENACC\n#ifdef __cplusplus\n}\n#endif\n#endif\n")

    # 4. Update init call TODO: Use a line number from the program
    source = buffer.translate()
    if force_soa:
        source = re.sub(r"\bop_init\b\s*\((.*)\)", "op_init_soa(\\1,1)", source)
        source = re.sub(r"\bop_mpi_init\b\s*\((.*)\)", "op_mpi_init_soa(\\1,1)", source)

    # Substitude the op_seq.h include for op_lib_cpp.h
    source = re.sub(r'#include\s+"op_seq\.h"', '#include "op_lib_cpp.h"', source)

    return source
