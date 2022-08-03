import copy
import io
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f2003
import pcpp
from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import Base, _set_parent

import fortran.parser
import fortran.translator.program
import op as OP
from language import Lang
from store import Kernel, Location, ParseError, Program


def base_deepcopy(self, memo):
    cls = self.__class__
    result = object.__new__(cls)

    memo[id(self)] = result

    for k, v in self.__dict__.items():
        if k == "parent":
            continue

        setattr(result, k, copy.deepcopy(v, memo))

    if hasattr(result, "items"):
        _set_parent(result, result.items)

    return result


def file_reader_deepcopy(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)

    memo[id(self)] = result

    setattr(result, "source", None)
    setattr(result, "file", None)

    for k, v in self.__dict__.items():
        if hasattr(result, k):
            continue

        setattr(result, k, copy.deepcopy(v, memo))

    return result


# Patch the fparser2 Base class to allow deepcopies
setattr(Base, "__deepcopy__", base_deepcopy)
setattr(FortranFileReader, "__deepcopy__", file_reader_deepcopy)


class Preprocessor(pcpp.Preprocessor):
    def __init__(self, lexer=None):
        super(Preprocessor, self).__init__(lexer)
        self.line_directive = None

    def on_comment(self, tok: str) -> bool:
        return True

    def on_error(self, file: str, line: int, msg: str) -> None:
        loc = Location(file, line, 0)
        raise ParseError(msg, loc)

    def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath) -> None:
        if is_system_include:
            raise pcpp.OutputDirective(pcpp.Action.IgnoreAndPassThrough)

        super.on_include_not_found(is_malformed, is_system_include, curdir, includepath)


class Fortran(Lang):
    name = "Fortran"

    source_exts = ["F90", "F95"]
    include_ext = "inc"
    kernel_dir = True

    com_delim = "!"
    zero_idx = False

    def parseFile(
        self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]
    ) -> Tuple[f2003.Program, str]:
        preprocessor = Preprocessor()

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

        return parser(reader), source

    def parseProgram(self, path: Path, include_dirs: Set[Path], defines: List[str]) -> Program:
        ast, source = self.parseFile(path, frozenset(include_dirs), frozenset(defines))
        return fortran.parser.parseProgram(ast, source, path)

    def parseKernel(self, path: Path, name: str, include_dirs: Set[Path], defines: List[str]) -> Optional[Kernel]:
        ast, _ = self.parseFile(path, frozenset(include_dirs), frozenset(defines))
        return fortran.parser.parseKernel(ast, name, path)

    def translateProgram(self, program: Program, include_dirs: Set[Path], defines: List[str], force_soa: bool) -> str:
        ast, _ = self.parseFile(program.path, frozenset(include_dirs), frozenset(defines))
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
        elif isinstance(typ, OP.Custom):
            return typ.name
        else:
            assert False


Lang.register(Fortran)

import fortran.schemes
