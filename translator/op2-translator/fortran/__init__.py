from pathlib import Path
from types import MethodType
from typing import Optional, Set

import fortran.parser
import fortran.translator.program
import op as OP
from language import Lang
from store import Kernel, Program


class Fortran(Lang):
    name = "Fortran"

    source_exts = ["F90", "F95"]
    include_ext = "inc"
    kernel_dir = False

    com_delim = "!"
    zero_idx = False

    def parseProgram(self, path: Path, include_dirs: Set[Path]) -> Program:
        return fortran.parser.parseProgram(path, include_dirs)

    def parseKernel(self, path: Path, name: str, include_dirs: Set[Path]) -> Optional[Kernel]:
        return fortran.parser.parseKernel(path, name, include_dirs)

    def translateProgram(self, source: str, program: Program, force_soa: bool) -> str:
        return fortran.translator.program.translateProgram(source, program, force_soa)

    def formatType(self, typ: OP.Type) -> str:
        if isinstance(typ, OP.Int):
            if not typ.signed:
                raise NotImplementedError("Fortran does not support unsigned integers")

            return f"INTEGER(kind={int(typ.size / 8)})"
        elif isinstance(typ, OP.Float):
            return f"REAL(kind={int(typ.size / 8)})"
        elif isinstance(typ, OP.Bool):
            return f"LOGICAL"
        else:
            assert False


Lang.register(Fortran)
