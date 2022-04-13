import os
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import FrozenSet, Optional, Set

import clang.cindex
from dotenv import load_dotenv

import cpp.parser
import cpp.translator.program
import op as OP
from language import Lang
from store import Kernel, ParseError, Program

# Load environment vairables set in .env and set libclang path
load_dotenv()
clang.cindex.Config.set_library_file(os.getenv("LIBCLANG_PATH"))


class Cpp(Lang):
    name = "C++"

    source_exts = ["cpp"]
    include_ext = "h"
    kernel_dir = True

    com_delim = "//"
    zero_idx = True

    @lru_cache(maxsize=None)
    def parseFile(self, path: Path, include_dirs: FrozenSet[Path]) -> clang.cindex.TranslationUnit:
        args = [f"-I{dir}" for dir in include_dirs]
        translation_unit = clang.cindex.Index.create().parse(
            path, args=args, options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )

        error = next(iter(translation_unit.diagnostics), None)
        if error:
            raise ParseError(error.spelling, cpp.parser.parseLocation(error))

        return translation_unit

    def parseProgram(self, path: Path, include_dirs: Set[Path]) -> Program:
        return cpp.parser.parseProgram(self.parseFile(path, frozenset(include_dirs)), path)

    def parseKernel(self, path: Path, name: str, include_dirs: Set[Path]) -> Optional[Kernel]:
        return cpp.parser.parseKernel(self.parseFile(path, frozenset(include_dirs)), name, path)

    def translateProgram(self, program: Program, include_dirs: Set[Path], force_soa: bool) -> str:
        return cpp.translator.program.translateProgram(program.path.read_text(), program, force_soa)

    def formatType(self, typ: OP.Type) -> str:
        int_types = {
            (True, 32): "int",
            (True, 64): "long long",
            (False, 32): "unsigned",
            (False, 64): "unsigned long long",
        }

        float_types = {32: "float", 64: "double"}

        if isinstance(typ, OP.Int):
            return int_types[(typ.signed, typ.size)]
        elif isinstance(typ, OP.Float):
            return float_types[typ.size]
        elif isinstance(typ, OP.Bool):
            return "bool"
        else:
            assert False


Lang.register(Cpp)
