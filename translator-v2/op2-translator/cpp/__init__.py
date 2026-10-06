import logging
import os
import importlib
import subprocess
from argparse import ArgumentParser, Namespace
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any

import pcpp

import op as OP
from language import Lang
from store import Application, Location, ParseError, Program

logger = logging.getLogger(__name__)

SYSTEM_INCLUDES = None


class Preprocessor(pcpp.Preprocessor):
    def __init__(self, lexer=None):
        super(Preprocessor, self).__init__(lexer)
        self.line_directive = None

    def on_error(self, file: str, line: int, msg: str) -> None:
        loc = Location(file, line, 0)
        raise ParseError(msg, loc)

    def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath) -> None:
        if is_system_include:
            raise pcpp.OutputDirective(pcpp.Action.IgnoreAndPassThrough)

        super().on_include_not_found(is_malformed, is_system_include, curdir, includepath)


class Cpp(Lang):
    name = "C++"

    source_exts = ["cpp"]
    include_ext = "h"

    com_delim = "//"
    ast_is_serializable = False

    fallback_wrapper_template = None

    def addArgs(self, parser: ArgumentParser) -> None:
        pass

    def parseArgs(self, args: Namespace) -> None:
        pass

    def validate(self, app: Application) -> None:
        pass

    @lru_cache(maxsize=None)
    def parseFile(
        self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str], preprocess: bool = False
    ) -> Tuple[Any, str]:
        import cpp.parser

        global SYSTEM_INCLUDES

        # Query system compiler for include dirs - should work as long as GCC/Clang turns up as "c++"
        if SYSTEM_INCLUDES is None:
            res = subprocess.run(["c++", "-xc++", "/dev/null", "-E", "-Wp,-v"],
                                 stderr=subprocess.PIPE,
                                 stdout=subprocess.PIPE)

            output = res.stderr.decode('utf-8').splitlines()
            SYSTEM_INCLUDES = [path[1:] for path in output if path.startswith(" ")]

        args = [f"-isystem{dir}" for dir in SYSTEM_INCLUDES]
        args = args + [f"-I{dir}" for dir in include_dirs]
        args = args + [f"-D{define}" for define in defines]

        source = path.read_text()
        if preprocess:
            preprocessor = Preprocessor()

            for dir in include_dirs:
                preprocessor.add_path(str(dir.resolve()))

            for define in defines:
                if "=" not in define:
                    define = f"{define}=1"

                preprocessor.define(define.replace("=", " ", 1))

            preprocessor.parse(source, str(path.resolve()))
            source_io = StringIO()
            preprocessor.write(source_io)

            source_io.seek(0)
            source = source_io.read()

        translation_unit = cpp.parser.parseTranslationUnit(path, source, args)

        return translation_unit, source

    def resolvedIncludes(self, translation_unit: Any) -> Set[Path]:
        # Headers reached from this translation unit, for depfile emission.
        # Toolchain headers are dropped the way `gcc -MM` drops them: they are
        # the dirs we passed as -isystem above, and tying retranslation to a
        # compiler update buys nothing. Only includes clang actually resolved
        # appear here - an unresolvable one is a build failure regardless.
        system_dirs = [str(Path(dir).resolve()) + os.sep for dir in (SYSTEM_INCLUDES or [])]

        includes: Set[Path] = set()
        for inclusion in translation_unit.get_includes():
            if inclusion.include is None:
                continue

            resolved = Path(inclusion.include.name).resolve()
            if any(str(resolved).startswith(dir) for dir in system_dirs):
                continue

            includes.add(resolved)

        return includes

    def parseProgram(self, path: Path, include_dirs: Set[Path], defines: List[str]) -> Program:
        import cpp.parser

        ast, source = self.parseFile(path, frozenset(include_dirs), frozenset(defines))
        ast_pp, source_pp = self.parseFile(path, frozenset(include_dirs), frozenset(defines), preprocess=True)

        program = Program(path, ast_pp, source_pp)

        # From the unpreprocessed parse: pcpp has already expanded the includes
        # out of the preprocessed one.
        program.includes = self.resolvedIncludes(ast)

        cpp.parser.parseLoops(ast, program)
        cpp.parser.parseMeta(ast_pp.cursor, program)

        cpp.parser.findLoopConsts(program)
        
        return program

    def translateProgram(self, program: Program, include_dirs: Set[Path], defines: List[str], force_soa: bool) -> str:
        import cpp.translator.program
        return cpp.translator.program.translateProgram(program.path.read_text(), program, force_soa)

    def formatType(self, typ: OP.Type) -> str:
        int_types: Dict[Tuple[bool, int], str] = {
            (True, 32): "int",
            (True, 64): "int64_t",
            (False, 32): "unsigned",
            (False, 64): "uint64_t",
        }

        float_types = {32: "float", 64: "double"}

        if isinstance(typ, OP.Int):
            return int_types[(typ.signed, typ.size)]
        elif isinstance(typ, OP.Float):
            return float_types[typ.size]
        elif isinstance(typ, OP.Bool):
            return "bool"
        elif isinstance(typ, OP.Custom):
            return typ.name
        else:
            assert False


Lang.register(Cpp)

import cpp.schemes
