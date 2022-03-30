import re

from store import Program
from util import SourceBuffer, safeFind


# Augment source program to use generated kernel hosts
def translateProgram(source: str, program: Program, force_soa: bool) -> str:
    buffer = SourceBuffer(source)

    # 1. Comment-out const calls
    for const in program.consts:
        buffer.apply(const.loc.line - 1, lambda line: "! " + line)

    # 2. Update loop calls
    for loop in program.loops:
        before, after = re.split(r"op_par_loop_[1-9]\d*", buffer.get(loop.loc.line - 1), 1)
        after = after.replace(
            loop.kernel, f'"{loop.kernel}"'
        )  # TODO: This assumes that the kernel arg is on the same line as the call
        buffer.update(loop.loc.line - 1, before + f"{loop.kernel}_host" + after)

    # 3. Update headers
    index = buffer.search(r"\s*use\s+OP2_Fortran_Reference\s*", re.IGNORECASE)
    buffer.apply(index, lambda line: "! " + line)
    for loop in program.loops:
        buffer.insert(index, f"  use {loop.kernel.upper()}_MODULE")

    # 4. Update init call TODO: Use a line number from the program
    source = buffer.translate()
    if force_soa:
        source = re.sub(r"\bop_init(\w*)\b\s*\((.*)\)", "op_init\\1_soa(\\2,1)", source)
        source = re.sub(r"\bop_mpi_init(\w*)\b\s*\((.*)\)", "op_mpi_init\\1_soa(\\2,1)", source)

    return source