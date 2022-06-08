from __future__ import annotations

from dataclasses import dataclass, field
from os.path import basename
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
from textwrap import indent

from typing_extensions import Protocol

import op as OP
from op import OpError
from util import find, flatten, safeFind, uniqueBy

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
class Program:
    path: Path

    consts: List[OP.Const] = field(default_factory=list)
    loops: List[OP.Loop] = field(default_factory=list)

    def __str__(self) -> str:
        consts_str = "\n    ".join([str(c) for c in self.consts])
        loops_str = "\n".join([str(l) for l in self.loops])

        if len(self.consts) > 0:
            consts_str = f"    {consts_str}\n"

        if len(self.loops) > 0:
            loops_str = indent(f"\n{loops_str}", "    ")

        return f"Program in '{self.path}':\n" + consts_str + loops_str


@dataclass
class Kernel:
    name: str
    path: Path

    params: List[Tuple[str, OP.Type]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Kernel in '{self.path}':\n"  # fmt: skip
            f"    {self.name}({', '.join([f'{p[0]}: {repr(p[1])}' for p in self.params])})\n"
        )


@dataclass
class Application:
    programs: List[Program] = field(default_factory=list)
    kernels: Dict[str, Kernel] = field(default_factory=dict)

    def __str__(self) -> str:
        if len(self.programs) > 0:
            programs_str = "\n".join([str(p) for p in self.programs])
        else:
            programs_str = "No programs"

        if len(self.kernels) > 0:
            kernels_str = "\n".join([str(k) for k in self.kernels.values()])
        else:
            kernels_str = "No kernels"

        return programs_str + "\n" + kernels_str

    def consts(self) -> List[OP.Const]:
        consts = flatten(program.consts for program in self.programs)
        return uniqueBy(consts, lambda c: c.ptr)

    def loops(self) -> List[OP.Loop]:
        loops = flatten(program.loops for program in self.programs)
        return uniqueBy(loops, lambda l: l.kernel)

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
        for loop in self.loops():
            num_opts = len([arg for arg in loop.args if arg.opt])
            if num_opts > 32:
                raise OpError(f"number of optional arguments exceeds 32: {num_opts}", loop.loc)

            for arg in loop.args:
                if isinstance(arg, OP.ArgDat):
                    self.validateArgDat(arg, loop, lang)

                if isinstance(arg, OP.ArgGbl):
                    self.validateArgGbl(arg, loop, lang)

            self.validateKernel(loop, lang)

    def validateArgDat(self, arg: OP.ArgDat, loop: OP.Loop, lang: Lang) -> None:
        valid_access_types = [OP.AccessType.READ, OP.AccessType.WRITE, OP.AccessType.RW, OP.AccessType.INC]
        if arg.access_type not in valid_access_types:
            raise OpError(f"invalid access type for dat argument: {arg.access_type}", arg.loc)

    def validateArgGbl(self, arg: OP.ArgGbl, loop: OP.Loop, lang: Lang) -> None:
        valid_access_types = [OP.AccessType.READ, OP.AccessType.INC, OP.AccessType.MIN, OP.AccessType.MAX]
        if arg.access_type not in valid_access_types:
            raise OpError(f"invalid access type for gbl argument: {arg.access_type}", arg.loc)

        if arg.access_type != OP.AccessType.READ and arg.typ not in [OP.Float(64), OP.Float(32), OP.Int(True, 32)]:
            raise OpError(f"invalid access type for reduced gbl argument: {arg.access_type}", arg.loc)

        if arg.dim < 1:
            raise OpError(f"invalid gbl argument dimension: {arg.dim}", arg.loc)

    def validateKernel(self, loop: OP.Loop, lang: Lang) -> None:
        kernel = self.kernels[loop.kernel]

        if len(loop.args) != len(kernel.params):
            raise OpError(f"number of loop arguments does not match number of kernel arguments", loop.loc)

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
