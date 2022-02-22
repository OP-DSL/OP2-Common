import os
from pathlib import Path
from types import MethodType
from typing import Set

import clang.cindex
from dotenv import load_dotenv

import cpp.parser
import cpp.translator.program
import op as OP
from language import Lang
from store import Kernel, Program

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

    def parseProgram(self, path: Path, include_dirs: Set[Path]) -> Program:
        return cpp.parser.parseProgram(path, include_dirs)

    def parseKernel(self, path: Path, name: str) -> Kernel:
        return cpp.parser.parseKernel(path, name)

    def translateProgram(self, source: str, program: Program) -> str:
        return cpp.translator.program.translateProgram(source, program)

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
