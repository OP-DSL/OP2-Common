from __future__ import annotations

import copy
from dataclasses import dataclass, field
from os.path import basename
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import op as OP
from op import OpError
from util import flatten, uniqueBy

if TYPE_CHECKING:
    from language import Lang


@dataclass(frozen=True)
class Location:
    file: str
    line: int
    column: int

    def __str__(self) -> str:
        return f"{basename(self.file)}/{self.line}:{self.column}"


@dataclass
class ParseError(Exception):
    message: str
    loc: Optional[Location] = None

    def __str__(self) -> str:
        if self.loc:
            return f"Parse error at {self.loc}: {self.message}"
        else:
            return f"Parse error: {self.message}"


@dataclass
class Entity:
    name: str
    ast: Any

    program: Program
    scope: List[str] = field(default_factory=list)
    depends: Set[str] = field(default_factory=set)

    # Deep-copy everything but the program reference
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)

        setattr(result, "program", self.program)

        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if hasattr(result, k):
                continue

            setattr(result, k, copy.deepcopy(v, memo))

        return result


@dataclass
class Type(Entity):
    def __str__(self):
        return f"Type(name='{self.name}', scope={self.scope}, depends={self.depends})"


@dataclass
class Function(Entity):
    parameters: List[str] = field(default_factory=list)
    returns: Optional[OP.Type] = None

    def __str__(self):
        return f"Function(name='{self.name}', scope={self.scope}, depends={self.depends})"


@dataclass
class Program:
    path: Path

    ast: Any
    source: str

    consts: List[OP.Const] = field(default_factory=list)
    loops: List[OP.Loop] = field(default_factory=list)

    entities: List[Entity] = field(default_factory=list)

    def findEntities(self, name: str, scope: List[str] = []) -> List[Entity]:
        def in_scope(e):
            return len(e.scope) <= len(scope) and all(map(lambda s1, s2: s1 == s2, zip(e.scope, scope)))

        candidates = list(filter(lambda e: e.name == name and in_scope(e), self.entities))
        if len(candidates) == 0:
            return []

        candidates.sort(key=lambda e: len(e.scope), reverse=True)
        min_scope = len(candidates[0].scope)

        return list(filter(lambda e: len(e.scope) == min_scope, candidates))

    def __str__(self) -> str:
        consts_str = "\n    ".join([str(const) for const in self.consts])
        loops_str = "\n".join([str(loop) for loop in self.loops])
        entities_str = "\n".join([str(entity) for entity in self.entities])

        if len(self.consts) > 0:
            consts_str = f"    {consts_str}\n"

        if len(self.loops) > 0:
            loops_str = indent(f"\n{loops_str}", "    ")

        if len(self.entities) > 0:
            entities_str = indent(f"\n{entities_str}\n", "    ")

        return f"Program in '{self.path}':\n" + consts_str + loops_str + entities_str


@dataclass
class Application:
    programs: List[Program] = field(default_factory=list)

    def __str__(self) -> str:
        if len(self.programs) > 0:
            programs_str = "\n".join([str(p) for p in self.programs])
        else:
            programs_str = "No programs"

        return programs_str

    def findEntities(self, name: str, program: Program = None, scope: List[str] = []) -> List[Entity]:
        candidates = []

        if program is not None:
            candidates = program.findEntities(name, scope)

        if len(candidates) > 0:
            return candidates

        for program2 in self.programs:
            if program2 == program:
                continue

            candidates = program2.findEntities(name)
            if len(candidates) > 0:
                break

        return candidates

    def consts(self) -> List[OP.Const]:
        consts = flatten(program.consts for program in self.programs)
        return uniqueBy(consts, lambda c: c.ptr)

    def loops(self) -> List[Tuple[OP.Loop, Program]]:
        loops = flatten(map(lambda l: (l, p), p.loops) for p in self.programs)
        return uniqueBy(loops, lambda l: l[0].kernel)

    def validate(self, lang: Lang) -> None:
        self.validateConsts(lang)
        self.validateLoops(lang)

    def validateConsts(self, lang: Lang) -> None:
        seen_const_ptrs: Set[str] = set()

        for const in self.consts():
            if const.ptr in seen_const_ptrs:
                raise OpError(f"duplicate const declaration: {const.ptr}", const.loc)

            seen_const_ptrs.add(const.ptr)

            if const.dim < 1:
                raise OpError(f"invalid const dimension: {const.dim}", const.dim)

    def validateLoops(self, lang: Lang) -> None:
        for loop, program in self.loops():
            num_opts = len([arg for arg in loop.args if arg.opt])
            if num_opts > 32:
                raise OpError(f"number of optional arguments exceeds 32: {num_opts}", loop.loc)

            for arg in loop.args:
                if isinstance(arg, OP.ArgDat):
                    self.validateArgDat(arg, loop, lang)

                if isinstance(arg, OP.ArgGbl):
                    self.validateArgGbl(arg, loop, lang)

            # self.validateKernel(loop, program, lang) TODO

    def validateArgDat(self, arg: OP.ArgDat, loop: OP.Loop, lang: Lang) -> None:
        valid_access_types = [OP.AccessType.READ, OP.AccessType.WRITE, OP.AccessType.RW, OP.AccessType.INC]
        if arg.access_type not in valid_access_types:
            raise OpError(f"invalid access type for dat argument: {arg.access_type}", arg.loc)

    def validateArgGbl(self, arg: OP.ArgGbl, loop: OP.Loop, lang: Lang) -> None:
        valid_access_types = [
            OP.AccessType.READ,
            OP.AccessType.WRITE,
            OP.AccessType.RW,
            OP.AccessType.INC,
            OP.AccessType.MIN,
            OP.AccessType.MAX,
        ]
        if arg.access_type not in valid_access_types:
            raise OpError(f"invalid access type for gbl argument: {arg.access_type}", arg.loc)

        if arg.access_type != OP.AccessType.READ and arg.typ not in [OP.Float(64), OP.Float(32), OP.Int(True, 32)]:
            raise OpError(f"invalid access type for reduced gbl argument: {arg.access_type}", arg.loc)

        if arg.dim < 1:
            raise OpError(f"invalid gbl argument dimension: {arg.dim}", arg.loc)

    # TODO: Re-do kernel validation
    def validateKernel(self, loop: OP.Loop, program: Program, lang: Lang) -> None:
        kernel_entities = self.findEntities(loop.kernel, program)  # TODO: Loop scope

        if len(loop.args) != len(kernel.params):
            raise OpError("number of loop arguments does not match number of kernel arguments", loop.loc)

        for loop_arg, kernel_param in zip(loop.args, kernel.params):
            if isinstance(loop_arg, OP.ArgDat) and loop.dats[loop_arg.dat_id].typ != kernel_param[1]:
                raise OpError(
                    f"loop argument type does not match kernel paramater type: {loop_arg.dat_typ} != {kernel_param[1]}",
                    loop.loc,
                )

            if isinstance(loop_arg, OP.ArgGbl) and loop_arg.typ != kernel_param[1]:
                raise OpError(
                    f"loop argument type does not match kernel paramater type: {loop_arg.typ} != {kernel_param[1]}",
                    loop.loc,
                )
