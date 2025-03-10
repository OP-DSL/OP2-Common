from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from util import ABDC, findIdx

if TYPE_CHECKING:
    from store import Location


class AccessType(Enum):
    READ = 0
    WRITE = 1
    RW = 2

    INC = 3
    MIN = 4
    MAX = 5

    WORK = 6

    @staticmethod
    def values() -> List[int]:
        return [x.value for x in list(AccessType)]

    @staticmethod
    def names() -> List[str]:
        return [x.name for x in list(AccessType)]


class OpError(Exception):
    message: str
    loc: Optional[Location]

    def __init__(self, message: str, loc: Optional[Location] = None) -> None:
        self.message = message
        self.loc = loc

    def __str__(self) -> str:
        if self.loc is not None:
            return f"{self.loc}: OP error: {self.message}"
        else:
            return f"OP error: {self.message}"


class Type:
    formatter: Callable[["Type"], str]

    @classmethod
    def set_formatter(cls, formatter: Callable[["Type"], str]) -> None:
        cls.formatter = formatter

    def c(self) -> str:
        raise OpError(f"Could not format {self} as a C type")

    def __str__(self) -> str:
        return self.__class__.formatter(self)


@dataclass(frozen=True)
class Int(Type):
    signed: bool
    size: int

    def c(self) -> str:
        int_types: Dict[Tuple[bool, int], str] = {
            (True, 32): "int",
            (True, 64): "int64_t",
            (False, 32): "unsigned",
            (False, 64): "uint64_t",
        }

        return int_types[(self.signed, self.size)]

    def __repr__(self) -> str:
        if self.signed and self.size == 32:
            return "int"
        elif self.size == 32:
            return "unsigned"
        else:
            return f"{'i' if self.signed else 'u'}{self.size}"


@dataclass(frozen=True)
class Float(Type):
    size: int

    def c(self) -> str:
        float_types = {32: "float", 64: "double"}
        return float_types[self.size]

    def __repr__(self) -> str:
        if self.size == 32:
            return "float"
        elif self.size == 64:
            return "double"
        else:
            return "f{self.size}"


@dataclass(frozen=True)
class Bool(Type):
    pass

    def c(self) -> str:
        return "bool"

    def __repr__(self) -> str:
        return "bool"


@dataclass(frozen=True)
class Custom(Type):
    name: str

    def c(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const:
    loc: Location
    ptr: str

    dim: int
    typ: Type

    def __str__(self) -> str:
        return f"Const(loc={self.loc}, ptr='{self.ptr}', dim={self.dim}, typ={self.typ})"


@dataclass(frozen=True)
class Map:
    id: int

    ptr: str
    arg_id: int


@dataclass(frozen=True)
class Dat:
    id: int

    ptr: str
    arg_id: int

    dim: Optional[int]
    typ: Type
    soa: bool

    def __str__(self) -> str:
        return (
            f"Dat(id={self.id}, ptr='{self.ptr}', arg_id={self.arg_id}, dim={self.dim}, typ={self.typ}, soa={self.soa})"
        )


@dataclass(frozen=True)
class Arg(ABDC):
    id: int
    loc: Location


@dataclass(frozen=True)
class ArgDat(Arg):
    access_type: AccessType
    opt: bool

    dat_id: int

    map_id: Optional[int]
    map_idx: Optional[int]

    def __str__(self) -> str:
        return (
            f"ArgDat(id={self.id}, loc={self.loc}, access_type={str(self.access_type) + ',':17} opt={self.opt}, "
            f"dat_id={self.dat_id}, map_id={self.map_id}, map_idx={self.map_idx})"
        )


@dataclass(frozen=True)
class ArgGbl(Arg):
    access_type: AccessType
    opt: bool

    ptr: str

    dim: Optional[int]
    typ: Type

    def __str__(self) -> str:
        return (
            f"ArgGbl(id={self.id}, loc={self.loc}, access_type={str(self.access_type) + ',':17} opt={self.opt}, "
            f"ptr={self.ptr}, dim={self.dim}, typ={self.typ})"
        )


@dataclass(frozen=True)
class ArgIdx(Arg):
    map_id: Optional[int]
    map_idx: Optional[int]

    def __str__(self) -> str:
        return f"ArgIdx(id={self.id}, loc={self.loc}, map_id={self.map_id}, map_idx={self.map_idx})"


@dataclass(frozen=True)
class ArgInfo(Arg):
    ptr: str

    dim: Optional[int]
    typ: Type

    ref: int

    def __str__(self) -> str:
        return f"ArgInfo(id={self.id}, loc={self.loc}, ptr={self.ptr}, dim={self.dim}, typ={self.typ}, ref={self.ref})"


class Loop:
    name: str

    loc: Location
    kernel: str

    args: List[Arg]
    args_expanded: List[Arg]

    dats: List[Dat]
    maps: List[Map]

    consts: Set[str]

    fallback: bool

    def __init__(self, name: str, loc: Location, kernel: str) -> None:
        self.name = name

        self.loc = loc
        self.kernel = kernel

        self.dats = []
        self.maps = []

        self.args = []
        self.args_expanded = []

        self.consts = set()

        self.fallback = False

    def addArgDat(
        self,
        loc: Location,
        dat_ptr: str,
        dat_dim: Optional[int],
        dat_typ: Type,
        dat_soa: bool,
        map_ptr: Optional[str],
        map_idx: Optional[int],
        access_type: AccessType,
        opt: bool = False,
    ) -> None:
        arg_id = len(self.args)

        dat_id = findIdx(self.dats, lambda d: d.ptr == dat_ptr)
        if dat_id is None:
            dat_id = len(self.dats)

            dat = Dat(dat_id, dat_ptr, arg_id, dat_dim, dat_typ, dat_soa)
            self.dats.append(dat)
        elif self.dats[dat_id].dim is None and dat_dim is not None:
            self.dats[dat_id] = dataclasses.replace(self.dats[dat_id], dim=dat_dim)

        map_id = None
        if map_ptr is not None:
            map_id = findIdx(self.maps, lambda m: m.ptr == map_ptr)

            if map_id is None:
                map_id = len(self.maps)

                map_ = Map(map_id, map_ptr, arg_id)
                self.maps.append(map_)

        arg = ArgDat(arg_id, loc, access_type, opt, dat_id, map_id, map_idx)
        self.args.append(arg)

        if map_ptr is None or map_idx is None or map_idx >= 0:
            self.args_expanded.append(arg)
            return

        for real_map_idx in range(-map_idx):
            arg_expanded = dataclasses.replace(arg, map_idx=real_map_idx)
            self.args_expanded.append(arg_expanded)

    def addArgGbl(
        self, loc: Location, ptr: str, dim: Optional[int], typ: Type, access_type: AccessType, opt: bool
    ) -> None:
        arg_id = len(self.args)
        arg = ArgGbl(arg_id, loc, access_type, opt, ptr, dim, typ)

        self.args.append(arg)
        self.args_expanded.append(arg)

    def addArgIdx(self, loc: Location, map_ptr: Optional[str], map_idx: Optional[int]) -> None:
        arg_id = len(self.args)

        map_id = None
        if map_ptr is not None:
            map_id = findIdx(self.maps, lambda m: m.ptr == map_ptr)

            if map_id is None:
                map_id = len(self.maps)

                map_ = Map(map_id, map_ptr, arg_id)
                self.maps.append(map_)

        arg = ArgIdx(arg_id, loc, map_id, map_idx)
        self.args.append(arg)

    def addArgInfo(self, loc: Location, ptr: str, dim: Optional[int], typ: Type, ref: int) -> None:
        arg_id = len(self.args)
        arg = ArgInfo(arg_id, loc, ptr, dim, typ, ref)

        self.args.append(arg)
        self.args_expanded.append(arg)

    def optIdx(self, arg: Arg) -> Optional[int]:
        idx = 0
        for arg2 in self.args:
            if arg2 == arg:
                break

            opt = getattr(arg2, "opt", False)
            if opt == True:
                idx += 1

        return idx

    def addConst(self, const: str) -> None:
        self.consts.add(const)

    def arg(self, x: Union[Dat, int]) -> Optional[Arg]:
        if isinstance(x, Dat):
            return self.args[x.arg_id]

        if isinstance(x, int) and x < len(self.args):
            return self.args[x]

        return None

    def dat(self, x: Union[ArgDat, int]) -> Optional[Dat]:
        if isinstance(x, ArgDat):
            return self.dats[x.dat_id]

        if isinstance(x, int) and x < len(self.dats):
            return self.dats[x]

        return None

    def map(self, x: Union[ArgDat, int]) -> Optional[Map]:
        if isinstance(x, ArgDat) and x.map_id is not None:
            return self.maps[x.map_id]

        if isinstance(x, int) and x < len(self.maps):
            return self.maps[x]

        return None

    def __str__(self) -> str:
        args = "\n    ".join([str(a) for a in self.args])

        dat_str = "\n    ".join([str(d) for d in self.dats])
        map_str = "\n    ".join([str(m) for m in self.maps])

        if len(self.dats) > 0:
            dat_str = f"\n    {dat_str}\n"

        if len(self.maps) > 0:
            map_str = f"\n    {map_str}\n"

        return (
            f"Loop at {self.loc}:\n    Name: {self.name}\n    Kernel function: {self.kernel}\n\n    {args}\n"
            + dat_str
            + map_str
        )
