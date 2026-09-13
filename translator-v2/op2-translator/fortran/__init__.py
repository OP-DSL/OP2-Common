import copy
import io
import logging
import os
import re
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import FrozenSet, List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f2003
import fparser.two.utils
import pcpp
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import Base, _set_parent

import fortran.flang_parser
import fortran.flang_validator
import fortran.fparser2_fallback
import fortran.parser
import fortran.translator.program
import fortran.validator
import op as OP
from language import Lang
from store import Application, Location, ParseError, Program

logger = logging.getLogger(__name__)


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


def string_reader_deepcopy(self, memo):
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
Base.__deepcopy__ = base_deepcopy  # type: ignore
FortranStringReader.__deepcopy__ = string_reader_deepcopy  # type: ignore


old_base_new = Base.__new__  # type: ignore

def base_new(cls, string, parent_cls=None, _deepcopy=None):
    if string is None:
        return object.__new__(cls)

    return old_base_new(cls, string, parent_cls=parent_cls)

Base.__new__ = base_new  # type: ignore


def base_getnewargsex(self):
    return ((None,), {})

Base.__getnewargs_ex__ = base_getnewargsex


kind_selector_aliases = {"*PS": "*8"}


def kind_selector_match(string):
    if string in kind_selector_aliases:
        string = kind_selector_aliases[string]

    return f2003.Kind_Selector.match_(string)  # type: ignore


f2003.Kind_Selector.match_ = f2003.Kind_Selector.match  # type: ignore
f2003.Kind_Selector.match = staticmethod(kind_selector_match)


# Patch the updated fparser2 walk function that visits tuples
# TODO: remove this when it has been included in an fparser release
def walk(node_list, types=None, indent=0, debug=False):
    local_list = []

    if not isinstance(node_list, (list, tuple)):
        node_list = [node_list]

    for child in node_list:
        if debug:
            if isinstance(child, str):
                logger.debug(indent * "  " + "child type = %s %r", type(child), child)
            else:
                logger.debug(indent * "  " + "child type = %s", type(child))
        if types is None or isinstance(child, types):
            local_list.append(child)
        # Recurse down
        if isinstance(child, Base):
            local_list += walk(child.children, types, indent + 1, debug)
        elif isinstance(child, tuple):
            for component in child:
                local_list += walk(component, types, indent + 1, debug)

    return local_list


fparser.two.utils.walk = walk


class FortranSyntaxError(Exception):
    def __init__(self, message, filename):
        super().__init__()

        self.message = message
        self.filename = filename

    def __reduce__(self):
        return (FortranSyntaxError, (self.message, self.filename))


class Preprocessor(pcpp.Preprocessor):
    def __init__(self, lexer=None):
        super(Preprocessor, self).__init__(lexer)

        self.line_directive = None
        self.includes: Set[Path] = set()

    def on_file_open(self, is_system_include, includepath):
        # Called once per candidate path, so record only after the base class
        # has opened it - a path that does not exist raises out of here and
        # pcpp moves on to the next include dir.
        handle = super(Preprocessor, self).on_file_open(is_system_include, includepath)
        self.includes.add(Path(includepath).resolve())

        return handle

    def on_comment(self, tok):
        return tok.type == self.t_COMMENT2

    def on_error(self, file, line, msg):
        loc = Location(file, line, 0)
        raise ParseError(msg, loc)

    def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath):
        if is_system_include:
            raise pcpp.OutputDirective(pcpp.Action.IgnoreAndPassThrough)

        super(Preprocessor, self).on_include_not_found(is_malformed, is_system_include, curdir, includepath)


class Fortran(Lang):
    name = "Fortran"

    source_exts = ["F90", "F95", "f90"]
    include_ext = "inc"

    com_delim = "!"
    ast_is_serializable = True

    fallback_wrapper_template = Path("fortran/fallback_wrapper.F90.jinja")

    consts_module = None
    consts_module_ast = None

    extra_consts_list = None
    user_consts_module = None
    use_regex_translator = False

    requested_parser = "fparser2"  # "fparser2" (default) or "flang"
    flang_scan_bin = None
    _include_dirs: Set[Path] = set()
    _defines: List[str] = []

    parser = None
    fpp = None

    # fparser2 does some dynamic class setup on parser creation, so make sure we always have one for kernel translation
    def __init__(self):
        self.parser = ParserFactory().create(std="f2008")

    def _ensure_ast_flang_programs(self, app: Application) -> None:
        for program in app.programs:
            if getattr(program, "used_parser", "fparser2") != "flang":
                continue
            if program.ast is not None:
                continue
            fortran.fparser2_fallback.ensure_fparser2_ast(
                self, program, self._include_dirs, self._defines
            )

    def addArgs(self, parser: ArgumentParser) -> None:
        parser.add_argument("--consts-module", help="(Fortran) Custom consts module")

        parser.add_argument("--extra-consts-list", help="(Fortran) Extra consts to rename in kernels", default=None)
        parser.add_argument("--user-consts-module", help="(Fortran) Use a custom consts module", default=None)
        parser.add_argument(
            "--regex-program-translator", help="(Fortran) Use the regex-based program translator", action="store_true"
        )
        parser.add_argument(
            "--parser",
            help="(Fortran) Parser pipeline to use for translation",
            choices=["fparser2", "flang"],
            default="fparser2",
        )
        parser.add_argument(
            "--flang-scan",
            help="(Fortran) Path to the op2-flang-scan binary (used with --parser flang)",
            default=None,
        )

    def parseArgs(self, args: Namespace) -> None:
        if args.consts_module is not None:
            self.consts_module = args.consts_module
            logger.debug(f"Using consts module: {self.consts_module}")

        if args.extra_consts_list is not None:
            self.extra_consts_list = args.extra_consts_list
            logger.debug(f"Using extra consts list: {self.extra_consts_list}")

        if args.user_consts_module is not None:
            self.user_consts_module = args.user_consts_module
            logger.debug(f"Using consts module: {self.user_consts_module}")

        if args.regex_program_translator:
            self.use_regex_translator = True
            logger.debug("Using regex program translator")
            if args.verbose:
                print("Using regex program translator")

        self.requested_parser = getattr(args, "parser", "fparser2")
        self.flang_scan_bin = getattr(args, "flang_scan", None)

        # used for programs/loops that use fparser2 fallback, so need an AST
        self._include_dirs = set(Path(d[0]) for d in getattr(args, "I", []))
        self._defines = [d[0] for d in getattr(args, "D", [])]

        if args.verbose:
            print(f"Requested Fortran parser: {self.requested_parser}")

        # fpp is a bundled binary, not a Python package - locate it relative
        # to this file's own install location (translator-v2/fpp/fpp
        # in-tree, libexec/op2/translator/fpp/fpp when installed), the same
        # pattern jinja.py uses for ../resources/templates.  Not relative to
        # sys.executable: that only worked when fpp was copied next to a
        # CMake-managed venv's python3, which no longer happens - Python is
        # now a found dependency, not one this project provisions.
        fpp = str(Path(__file__).resolve().parent.parent.parent / "fpp" / "fpp")
        if os.path.exists(fpp):
            self.fpp = fpp
            logger.debug(f"Using packaged fpp for Fortran parsing: {fpp}")
        else:
            logger.debug(f"Packaged fpp not found at {fpp} - falling back to pcpp for Fortran preprocessing")

    def validate(self, app: Application) -> None:
        # TODO: see fortran.parser

        if fortran.flang_parser.parsed_with_flang(app):
            fortran.flang_parser.resolve_flang_dependencies(app)

        for program in app.programs:
            if getattr(program, "used_parser", "fparser2") == "fparser2":
                fortran.parser.parseFunctionDependencies(program, app)

        for loop, program in app.loops():
            used_parser = getattr(program, "used_parser", "fparser2")

            # use the Flang validator whenever the kernel and all its dependencies were parsed by Flang
            if used_parser == "flang" and fortran.flang_validator.can_validate_with_flang(loop, program, app):
                fortran.flang_validator.validateLoop(loop, program, app)
                continue

            if used_parser == "flang":
                # ensure every program in the app has an fparser2 AST before falling back to fparser2
                self._ensure_ast_flang_programs(app)

            fortran.validator.validateLoop(loop, program, app)

    def fppIncludes(self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]) -> Set[Path]:
        # A second fpp pass: -M reports the files it read but suppresses the
        # preprocessed output, so it cannot be folded into the run that
        # produces the source. Its output is one '<object>: <dependency>' rule
        # per line, with paths relative to the working directory.
        args = [self.fpp, "-P", "-free", "-f90", "-M"]

        for dir in include_dirs:
            args.append(f"-I{dir}")

        for define in defines:
            args.append(f"-D{define}")

        args.append(str(path))

        res = subprocess.run(args, capture_output=True, check=True)

        includes: Set[Path] = set()
        for line in res.stdout.decode("utf-8").splitlines():
            _, separator, dependency = line.partition(":")

            if separator and dependency.strip():
                includes.add(Path(dependency.strip()).resolve())

        return includes

    def preprocess(
        self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]
    ) -> Tuple[str, Set[Path]]:
        if self.fpp:
            args = [self.fpp, "-P", "-free", "-f90"]

            for dir in include_dirs:
                args.append(f"-I{dir}")

            for define in defines:
                args.append(f"-D{define}")

            args.append(str(path))

            res = subprocess.run(args, capture_output=True, check=True)
            return res.stdout.decode("utf-8"), self.fppIncludes(path, include_dirs, defines)

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

        source = source.read()

        source = re.sub(r"__FILE__", f'"{path}"', source)
        source = re.sub(r"__LINE__", "0", source)

        return source, preprocessor.includes

    def parseFile(
        self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]
    ) -> Tuple[f2003.Program, str, Set[Path]]:
        source, includes = self.preprocess(path, include_dirs, defines)

        try:
            reader = FortranStringReader(source, include_dirs=list(include_dirs))
            ast = self.parser(reader)
        except fparser.two.utils.FortranSyntaxError as err:
            raise FortranSyntaxError(str(err), path.name)

        return ast, source, includes

    def parseProgram(self, path: Path, include_dirs: Set[Path], defines: List[str]) -> Program:
        source, includes = self.preprocess(path, frozenset(include_dirs), frozenset(defines))

        if self.requested_parser == "flang":
            try:
                scan_bin = fortran.flang_parser.resolve_scan_binary(self.flang_scan_bin)
                data = fortran.flang_parser.run_scan(
                    source, path, scan_bin, include_dirs=include_dirs
                )
                program = fortran.flang_parser.build_program_from_flang(path, source, data)
                program.includes = includes
                return program
            except ParseError as err:
                print(
                    f"Warning: Flang parse failed for {path}; "
                    f"falling back to fparser2: {err}",
                    file=sys.stderr,
                )

        return self.parseProgramFparser2(path, source, include_dirs, includes)

    def parseProgramFparser2(
        self,
        path: Path,
        source: str,
        include_dirs: Set[Path],
        includes: Optional[Set[Path]] = None,
    ) -> Program:
        try:
            reader = FortranStringReader(source, include_dirs=list(include_dirs))
            ast = self.parser(reader)
        except fparser.two.utils.FortranSyntaxError as err:
            raise FortranSyntaxError(str(err), path.name)

        program = fortran.parser.parseProgram(ast, source, path)
        setattr(program, "used_parser", "fparser2")
        program.includes = includes or set()
        return program

    def parsePrograms(
        self, paths: List[Path], include_dirs: Set[Path], defines: List[str]
    ) -> List[Program]:
        """
        Parse many Fortran sources into Programs (when using ``--parser flang``).

        Preprocesses every file then runs a single ``op2-flang-scan --batch``
        subprocess to only invoke LLVM load and process spawn once.
        Per-file Flang failures fall back to fparser2 individually.
        """
        if self.requested_parser != "flang" or not paths:
            return [self.parseProgram(p, include_dirs, defines) for p in paths]

        frozen_inc = frozenset(include_dirs)
        frozen_defs = frozenset(defines)
        prepared: List[Tuple[Path, str, Set[Path]]] = []
        for path in paths:
            source, includes = self.preprocess(path, frozen_inc, frozen_defs)
            prepared.append((path, source, includes))

        try:
            scan_bin = fortran.flang_parser.resolve_scan_binary(self.flang_scan_bin)
            scanned = fortran.flang_parser.run_scan_batch(
                [(path, source) for path, source, _ in prepared],
                scan_bin,
                include_dirs=include_dirs,
            )
        except ParseError as err:
            print(
                f"Warning: Flang batch scan failed; "
                f"falling back to per-file parsing: {err}",
                file=sys.stderr,
            )
            return [self.parseProgram(p, include_dirs, defines) for p in paths]

        programs: List[Program] = []
        for path, source, includes in prepared:
            data = scanned[path]
            if data.get("error"):
                print(
                    f"Warning: Flang parse failed for {path}; "
                    f"falling back to fparser2: {data['error']}",
                    file=sys.stderr,
                )
                programs.append(self.parseProgramFparser2(path, source, include_dirs, includes))
                continue
            try:
                program = fortran.flang_parser.build_program_from_flang(path, source, data)
                program.includes = includes
                programs.append(program)
            except ParseError as err:
                print(
                    f"Warning: Flang build failed for {path}; "
                    f"falling back to fparser2: {err}",
                    file=sys.stderr,
                )
                programs.append(self.parseProgramFparser2(path, source, include_dirs, includes))
        return programs

    def translateProgram(self, program: Program, include_dirs: Set[Path], defines: List[str], force_soa: bool) -> str:
        if getattr(program, "used_parser", "fparser2") == "flang":
            return fortran.translator.program.translateProgram2(program, force_soa)

        if self.use_regex_translator or program.ast is None:
            if program.ast is None and not self.use_regex_translator:
                print(
                    f"Warning: fparser2 AST unavailable for {program.path}; "
                    f"using regex program translator fallback.",
                    file=sys.stderr,
                )
            return fortran.translator.program.translateProgram2(program, force_soa)

        return fortran.translator.program.translateProgram(program, force_soa)

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
