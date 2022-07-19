import io
from pathlib import Path
from typing import FrozenSet, List, Optional, Set

import fparser.two.Fortran2003 as f2003
import pcpp
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
from fparser.two.symbol_table import SYMBOL_TABLES

import fortran.parser
import fortran.translator.program
import op as OP
from language import Lang
from store import Kernel, Program


class Fortran(Lang):
    name = "Fortran"

    source_exts = ["F90", "F95"]
    include_ext = "inc"
    kernel_dir = True

    com_delim = "!"
    zero_idx = False

    def parseFile(self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]) -> f2003.Program:
        preprocessor = pcpp.Preprocessor()
        preprocessor.line_directive = None

        for dir in include_dirs:
            preprocessor.add_path(str(dir.resolve()))

        for define in defines:
            if "=" not in define:
                define = f"{define}=1"

            preprocessor.define(define.replace("=", " ", 1))

        preprocessor.parse(path.read_text(), str(path))

        source = io.StringIO()
        source.name = str(path)

        preprocessor.write(source)

        source.seek(0)

        reader = FortranFileReader(source, include_dirs=list(include_dirs))
        parser = ParserFactory().create(std="f2003")

        return parser(reader)

    def parseProgram(self, path: Path, include_dirs: Set[Path], defines: List[str]) -> Program:
        return fortran.parser.parseProgram(self.parseFile(path, frozenset(include_dirs), frozenset(defines)), path)

    def parseKernel(self, path: Path, name: str, include_dirs: Set[Path], defines: List[str]) -> Optional[Kernel]:
        return fortran.parser.parseKernel(self.parseFile(path, frozenset(include_dirs), frozenset(defines)), name, path)

    def translateProgram(self, program: Program, include_dirs: Set[Path], defines: List[str], force_soa: bool) -> str:
        ast = self.parseFile(program.path, frozenset(include_dirs), frozenset(defines))
        return fortran.translator.program.translateProgram(ast, program, force_soa)

    def formatType(self, typ: OP.Type) -> str:
        if isinstance(typ, OP.Int):
            if not typ.signed:
                raise NotImplementedError("Fortran does not support unsigned integers")

            return f"integer({int(typ.size / 8)})"
        elif isinstance(typ, OP.Float):
            return f"real({int(typ.size / 8)})"
        elif isinstance(typ, OP.Bool):
            return "logical"
        else:
            assert False


Lang.register(Fortran)

import fortran.schemes
