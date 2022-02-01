from pathlib import Path
from types import MethodType
from typing import Set

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

    def parseProgram(self, path: Path, include_dirs: Set[Path], soa: bool = False) -> Program:
        return fortran.parser.parseProgram(path, include_dirs, soa)

    def parseKernel(self, path: Path, name: str) -> Kernel:
        return fortran.parser.parseKernel(path, name)

    def translateProgram(self, source: str, program: Program, soa: bool = False) -> str:
        return fortran.translator.program.translateProgram(source, program, soa)

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
