from __future__ import annotations

from dataclasses import dataclass, field
from os.path import basename
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple

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
    kernels: List[Kernel] = field(default_factory=list)

    def __str__(self) -> str:
        if len(self.programs) > 0:
            programs_str = "\n".join([str(p) for p in self.programs])
        else:
            programs_str = "No programs"

        if len(self.kernels) > 0:
            kernels_str = "\n".join([str(k) for k in self.kernels])
        else:
            kernels_str = "No kernels"

        return programs_str + "\n" + kernels_str

    @property
    def hasInit(self) -> bool:
        return any(program.init for program in self.programs)

    @property
    def hasExit(self) -> bool:
        return any(program.exit for program in self.programs)

    @property
    def sets(self) -> List[OP.Set]:
        return flatten(program.sets for program in self.programs)

    @property
    def maps(self) -> List[OP.Map]:
        return flatten(program.maps for program in self.programs)

    @property
    def datas(self) -> List[OP.Data]:
        return flatten(program.dats for program in self.programs)

    @property
    def consts(self) -> List[OP.Const]:
        consts = flatten(program.consts for program in self.programs)
        return uniqueBy(consts, lambda c: c.ptr)

    @property
    def loops(self) -> List[OP.Loop]:
        loops = flatten(program.loops for program in self.programs)
        return uniqueBy(loops, lambda l: l.kernel)

    def validate(self, lang: Lang) -> None:
        return


#    def validate(self, lang: Lang) -> None:
#        if not self.hasInit:
#            print("warning: no call to op_init found")
#
#        if not self.hasExit:
#            print("warning: no call to op_exit found")
#
#        # Collect the pointers of defined sets
#        set_ptrs = [s.ptr for s in self.sets]
#
#        # Validate data declerations
#        for data in self.datas:
#            # Validate set
#            if data.set not in set_ptrs:
#                raise OpError(
#                    f'undefined set "{data.set}" referenced in data decleration',
#                    data.loc,
#                )
#
#        # Validate map declerations
#        for map in self.maps:
#            # Validate both sets
#            for set_ in (map.from_set, map.to_set):
#                if set_ not in set_ptrs:
#                    raise OpError(f'undefined set "{set_}" referenced in map decleration', map.loc)
#
#        # Validate constant declerations
#        for const in self.consts:
#            # Search for previous decleration
#            prev = safeFind(self.consts, lambda c: c.ptr == const.ptr)
#
#            if prev and const.dim != prev.dim:
#                raise ParseError(f'dim mismatch in repeated decleration of "{const.ptr}" const')
#            elif prev and const.dim != prev.dim:
#                raise ParseError(f'size mismatch in repeated decleration of "{const.ptr}" const')
#
#        # Validate loop calls
#        for loop in self.loops:
#            kern = safeFind(self.kernels, lambda k: k.name == loop.kernel)
#            loop.kernelPath = str(kern.path)
#            prev = safeFind(self.loops, lambda l: l.kernel == loop.kernel)
#            if prev:
#                for i, (arg_a, arg_b) in enumerate(zip(prev.args, loop.args)):
#                    if arg_a.acc != arg_b.acc:
#                        raise ParseError(f"varying access types for arg {i} in {loop.kernel} par loops")
#                    # TODO: Consider more compatability issues
#
#            # Validate loop dataset
#            if loop.set not in set_ptrs:
#                raise OpError(f'undefined set "{loop.set}" referenced in par loop call', loop.loc)
#
#            # Validate loop args
#            for arg in loop.args:
#                if not arg.global_:
#                    # Look for the referenced data
#                    data_ = safeFind(self.datas, lambda d: d.ptr == arg.var)
#
#                    # Validate the data referenced in the arg
#                    if not data_:
#                        raise OpError(
#                            f'undefined data "{arg.var}" referenced in par loop arg',
#                            arg.loc,
#                        )
#                    elif arg.typ != data_.typ:
#                        raise OpError(
#                            f"type mismatch of par loop data, expected {data_.typ}",
#                            arg.loc,
#                        )
#                    elif arg.dim != data_.dim:
#                        raise OpError(
#                            f"dimension mismatch of par loop data, expected {data_.dim}",
#                            arg.loc,
#                        )
#
#                    # Validate direct args
#                    if arg.direct:
#                        # Validate index
#                        if arg.idx != -1:
#                            raise OpError(
#                                "incompatible index for direct access, expected -1",
#                                arg.loc,
#                            )
#                        # Check the dataset can be accessed directly
#                        if data_.set != loop.set:
#                            raise OpError(
#                                f'cannot directly access the "{arg.var}" dataset from the "{loop.set}" loop set',
#                                arg.loc,
#                            )
#
#                        # Check that the same dataset has not already been directly accessed
#                        if safeFind(loop.directs, lambda a: a is not arg and a.var == arg.var):
#                            raise OpError(
#                                f'duplicate direct accesses to the "{arg.var}" dataset in the same par loop',
#                                arg.loc,
#                            )
#
#                    # Validate indirect args
#                    elif arg.indirect:
#                        # Look for the referenced map decleration
#                        map_ = safeFind(self.maps, lambda m: m.ptr == arg.map)
#
#                        if not map_:
#                            raise OpError(
#                                f'undefined map "{arg.map}" referenced in par loop arg',
#                                arg.loc,
#                            )
#
#                        # Check that the mapping maps from the loop set
#                        if map_.from_set != loop.set:
#                            raise OpError(
#                                f'cannot apply the "{arg.map}" mapping to the "{loop.set}" loop set',
#                                arg.loc,
#                            )
#
#                        # Check that the mapping maps to the data set
#                        if map_.to_set != data_.set:
#                            raise OpError(
#                                f'cannot map to the "{arg.var}" dataset with the "{arg.map}" mapping',
#                                arg.loc,
#                            )
#
#                        # Determine the valid index range using the given language
#                        min_idx = 0 if lang.zero_idx else 1
#                        max_idx = map_.dim - 1 if lang.zero_idx else map_.dim
#
#                        # Adjust min index for vec args
#                        # TODO check how Fortran OP2 does vec args
#                        if arg.vector and lang.zero_idx:
#                            min_idx = -map_.dim
#
#                        # Perform range check
#                        if arg.idx is None or arg.idx < min_idx or arg.idx > max_idx:
#                            raise OpError(
#                                f"index {arg.idx} out of range, must be in the interval [{min_idx},{max_idx}]",
#                                arg.loc,
#                            )
#
#                    # Enforce unique data access
#                    for other in loop.args:
#                        if (
#                            other is not arg
#                            and other.var == arg.var
#                            and (other.idx == arg.idx and other.map == arg.map)
#                        ):
#                            raise OpError(f"duplicate data accesses in the same par loop", arg.loc)
#
#            # Validate par loop arguments against kernel parameters
#            kernel = find(self.kernels, lambda k: k.name == loop.kernel)
#
#            if len(loop.args) != kernel.paramCount:
#                raise ParseError(f"incorrect number of args passed to the {kernel} kernel", loop.loc)
#
#            for i, (param, arg) in enumerate(zip(kernel.params, loop.args)):
#                if not arg.vector and arg.typ != param[1]:
#                    raise ParseError(
#                        f"argument {i} to {kernel} kernel has incompatible type {arg.typ}, expected {param[1]}",
#                        arg.loc,
#                    )
#                elif arg.vector and arg.typ != param[1][:-2]:
#                    raise ParseError(
#                        f"argument {i} to {kernel} kernel has incompatible type {arg.typ}, expected {param[1][:-2]}",
#                        arg.loc,
#                    )
