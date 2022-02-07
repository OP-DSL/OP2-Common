from __future__ import annotations

from dataclasses import dataclass, field
from os.path import basename
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Dict

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

    init: Optional[Location] = None
    exit: bool = False

    consts: List[OP.Const] = field(default_factory=list)

    sets: List[OP.Set] = field(default_factory=list)
    dats: List[OP.Dat] = field(default_factory=list)
    maps: List[OP.Map] = field(default_factory=list)

    loops: List[OP.Loop] = field(default_factory=list)

    def recordInit(self, loc: Location) -> None:
        if self.init:
            raise ParseError("duplicate op_init call", loc)

        self.init = loc

    def recordExit(self) -> None:
        self.exit = True

    def __str__(self) -> str:
        return (
            f"Program '{self.path}':\n"
            f"    init: {self.init}, exit: {self.exit}\n"
            f"    consts: {', '.join([f'{c.ptr}[{c.dim}]' for c in self.consts])}\n"
            f"\n"
            f"    sets: {', '.join([s.ptr for s in self.sets])}\n"
            f"    dats: {', '.join([f'{d.ptr}[{d.dim}]:{d.set_ptr}' for d in self.dats])}\n"
            f"    maps: {', '.join([f'{m.ptr}[{m.dim}]:{m.from_set_ptr}->{m.to_set_ptr}' for m in self.maps])}\n"
            f"\n"
            f"    loops: {', '.join([f'{l.kernel}/{len(l.args)}:{l.set_ptr}' for l in self.loops])}\n"
        )


@dataclass
class Kernel:
    name: str
    path: Path
    params: List[Tuple[str, OP.Type]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Kernel '{self.path}':\n"  # fmt: skip
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

    def hasInit(self) -> bool:
        return any(program.init for program in self.programs)

    def hasExit(self) -> bool:
        return any(program.exit for program in self.programs)

    def sets(self) -> List[OP.Set]:
        return flatten(program.sets for program in self.programs)

    def maps(self) -> List[OP.Map]:
        return flatten(program.maps for program in self.programs)

    def dats(self) -> List[OP.Data]:
        return flatten(program.dats for program in self.programs)

    def consts(self) -> List[OP.Const]:
        consts = flatten(program.consts for program in self.programs)
        return uniqueBy(consts, lambda c: c.ptr)

    def loops(self) -> List[OP.Loop]:
        loops = flatten(program.loops for program in self.programs)
        return uniqueBy(loops, lambda l: l.kernel)

    def validate(self, lang: Lang) -> None:
        if not self.hasInit:
            print("warning: no call to op_init")

        if not self.hasExit:
            print("warning: no call to op_exit")

        self.validateConsts(lang)

        self.validateSets(lang)
        self.validateDats(lang)
        self.validateMaps(lang)

        self.validateLoops(lang)

    def validateConsts(self, lang: Lang) -> None:
        seen_const_ptrs: Set[str] = set()

        for const in self.consts():
            if const.ptr in seen_const_ptrs:
                raise OpError(f"duplicate const declaration: {const.ptr}", const.loc)

            seen_const_ptrs.add(const.ptr)

            if const.dim < 1:
                raise OpError(f"invalid const dimension: {const.dim}", const.dim)

    def validateSets(self, lang: Lang) -> None:
        seen_set_ptrs: Set[str] = set()

        for set_ in self.sets():
            if set_.ptr in seen_set_ptrs:
                raise OpError(f"duplicate set declaration: {set_.ptr}", set_.loc)

            seen_set_ptrs.add(set_.ptr)

    def validateDats(self, lang: Lang) -> None:
        set_ptrs = {set_.ptr for set_ in self.sets()}
        seen_dat_ptrs: Set[str] = set()

        for dat in self.dats():
            if dat.ptr in seen_dat_ptrs:
                raise OpError(f"duplicate dat declaration: {dat.ptr}", dat.loc)

            seen_dat_ptrs.add(dat.ptr)

            if dat.dim < 1:
                raise OpError(f"invalid dat dimension: {dat.dim}", dat.loc)

            if dat.set_ptr not in set_ptrs:
                raise OpError(f"dat declaration references unknown set: {dat.set_ptr}", dat.loc)

    def validateMaps(self, lang: Lang) -> None:
        set_ptrs = {set_.ptr for set_ in self.sets()}
        seen_map_ptrs: Set[str] = set()

        for map_ in self.maps():
            if map_.ptr in seen_map_ptrs:
                raise OpError(f"duplicate map declaration: {map_.ptr}", map_.loc)

            seen_map_ptrs.add(map_.ptr)

            if map_.dim < 1:
                raise OpError(f"invalid map dimension: {map_.dim}", map_.loc)

            if map_.from_set_ptr not in set_ptrs:
                raise OpError(f"map declaration references unknown source set: {map_.from_set_ptr}", map_.loc)

            if map_.to_set_ptr not in set_ptrs:
                raise OpError(f"map declaration references unknown target set: {map_.to_set_ptr}", map_.loc)

    def validateLoops(self, lang: Lang) -> None:
        set_ptrs = {set_.ptr for set_ in self.sets()}

        for loop in self.loops():
            if loop.set_ptr not in set_ptrs:
                raise OpError(f"loop references unknown set: {loop.set_ptr}", loop.loc)

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

        dat = safeFind(self.dats(), lambda d: d.ptr == arg.dat_ptr)

        if dat is None:
            raise OpError(f"loop argument references unknown dat: {arg.dat_ptr}", arg.loc)

        if arg.dat_dim != dat.dim:
            raise OpError(f"loop argument dat dimension mismatch: {arg.dat_dim} (expected {dat.dim})", arg.loc)

        if arg.dat_typ != dat.typ:
            raise OpError(f"loop argument dat type mismatch: {arg.dat_typ} (expected {dat.typ})", arg.loc)

        if arg.map_ptr is None:
            if dat.set_ptr != loop.set_ptr:
                raise OpError(f"indirect loop argument dat requires a map: {arg.dat_ptr}", arg.loc)

            return

        map_ = safeFind(self.maps(), lambda m: m.ptr == arg.map_ptr)

        if map_ is None:
            raise OpError(f"loop argument references unknown map: {arg.map_ptr}", arg.loc)

        if arg.map_idx is None:
            raise OpError(f"indirect loop argument requires map index", arg.loc)

        if map_.from_set_ptr != loop.set_ptr:
            raise OpError(
                f"loop argument map from set mismatch: {map_.from_set_ptr} (expected {loop.set_ptr})", arg.loc
            )

        if map_.to_set_ptr != dat.set_ptr:
            raise OpError(f"loop argument map to set mismatch: {map_.to_set_ptr} (expected {dat.set_ptr})", arg.loc)


        idx_high = map_.dim if lang.zero_idx else map_.dim + 1
        if arg.map_idx < -map_.dim or arg.map_idx >= idx_high:
            raise OpError(
                f"loop argument map index out of range: {arg.map_idx} (expected {-map_.dim} <= idx < {idx_high})",
                arg.loc,
            )

        if not lang.zero_idx and arg.map_idx == 0:
            raise OpError(f"loop argument map index cannot be zero", arg.loc)

    def validateArgGbl(self, arg: OP.ArgGbl, loop: OP.Loop, lang: Lang) -> None:
        valid_access_types = [OP.AccessType.READ, OP.AccessType.INC, OP.AccessType.MIN, OP.AccessType.MAX]
        if arg.access_type not in valid_access_types:
            raise OpError(f"invalid access type for gbl argument: {arg.access_type}", arg.loc)

        if arg.dim < 1:
            raise OpError(f"invalid gbl argument dimension: {arg.dim}", arg.loc)

    def validateKernel(self, loop: OP.Loop, lang: Lang) -> None:
        kernel = self.kernels[loop.kernel]

        if len(loop.args) != len(kernel.params):
            raise OpError(f"number of loop arguments does not match number of kernel arguments", loop.loc)

        for loop_arg, kernel_param in zip(loop.args, kernel.params):
            if isinstance(loop_arg, OP.ArgDat) and loop_arg.dat_typ != kernel_param[1]:
                raise OpError(
                    f"loop argument type does not match kernel paramater type: {loop_arg.dat_typ} != {kernel_param[1]}",
                    loop.loc,
                )

            if isinstance(loop_arg, OP.ArgGbl) and loop_arg.typ != kernel_param[1]:
                raise OpError(
                    f"loop argument type does not match kernel paramater type: {loop_arg.typ} != {kernel_param[1]}",
                    loop.loc,
                )
