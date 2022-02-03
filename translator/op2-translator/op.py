from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, Final, List, Optional, Union

from cached_property import cached_property

from util import find, uniqueBy

if TYPE_CHECKING:
    from store import Location


class AccessType(Enum):
    ID = "OP_ID"

    READ = "OP_READ"
    WRITE = "OP_WRITE"
    RW = "OP_RW"

    INC = "OP_INC"
    MIN = "OP_MIN"
    MAX = "OP_MAX"

    @staticmethod
    def validDatTypes() -> List[AccessType]:
        return [AccessType.READ, AccessType.WRITE, AccessType.RW, AccessType.INC]

    @staticmethod
    def validGblTypes() -> List[AccessType]:
        return [AccessType.READ, AccessType.INC, AccessType.MAX, AccessType.MIN]


class OpError(Exception):
    message: str
    loc: Location

    def __init__(self, message: str, loc: Location = None) -> None:
        self.message = message
        self.loc = loc

    def __str__(self) -> str:
        if self.loc:
            return f"{self.loc}: OP error: {self.message}"
        else:
            return f"OP error: {self.message}"


class Type:
    formatter: Callable[["Type"], str]

    @classmethod
    def set_formatter(cls, formatter: Callable[["Type"], str]) -> None:
        cls.formatter = formatter

    def __str__(self) -> str:
        return self.__class__.formatter(self)


@dataclass
class Int(Type):
    signed: bool
    size: int

    def __repr__(self) -> str:
        if self.signed:
            return f"i{self.size}"
        else:
            return f"u{self.size}"


@dataclass
class Float(Type):
    size: int

    def __repr__(self) -> str:
        return f"f{self.size}"


@dataclass
class Bool(Type):
    pass

    def __repr__(self) -> str:
        return "bool"


@dataclass
class Set:
    loc: Location
    ptr: str


@dataclass
class Map:
    loc: Location

    from_set_ptr: str
    to_set_ptr: str

    dim: int
    ptr: str


@dataclass
class Dat:
    loc: Location

    set_ptr: str

    dim: int
    typ: Type
    ptr: str


@dataclass
class Const:
    loc: Location

    dim: int
    typ: Type
    ptr: str


@dataclass
class Arg:
    loc: Location

    dat_ptr: str
    dat_dim: int
    dat_typ: Type

    access_type: AccessType

    map_ptr: Optional[str] = None
    map_idx: Optional[int] = None

    opt: Optional[str] = None

    def isDirect(self) -> bool:
        return self.map_ptr == "OP_ID"

    def isIndirect(self) -> bool:
        return not self.isDirect()

    def isGbl(self) -> bool:
        return self.map_ptr is None

    def isVec(self) -> bool:
        return self.map_idx is not None and self.map_idx < 0


class Loop:
    loc: Location

    kernel: str
    set_ptr: str
    args: List[Arg]

    def __init__(self, loc: Location, kernel: str, set_ptr: str, args: List[Arg] = []) -> None:
        self.loc = loc
        self.kernel = kernel
        self.set_ptr = set_ptr
        self.args = []

        for arg in args:
            self.addArg(arg)

    def addArg(self, arg: Arg) -> None:
        self.args.append(arg)


# class Loop:
#     loc: Location
#     kernel: str
#     kernelPath: str
#
#     set: str
#     args: List[Arg]
#     expanded_args: List[Arg]
#
#     nargs: int
#     ninds: int
#
#     def __init__(self, kernel: str, set_: str, loc: Location, args: List[Arg]) -> None:
#         self.kernel = kernel
#         self.set = set_
#         self.loc = loc
#         self.args = args
#         self.expanded_args = []
#
#         ind_i = 0
#         for i, arg in enumerate(self.args):
#             arg.i = i
#             if arg.indirect:
#                 arg.indI = ind_i
#                 ind_i += 1
#         self.ninds = ind_i
#
#         # Number of arguments for this loop if no vec arguments
#         self.nargs_novec = len(self.args)
#
#         # Create a new list of args by expanding the vec args
#         for arg in self.args:
#             if arg.idx is not None and arg.idx < 0 and arg.indirect:
#                 # Expand vec arg into multiple args
#                 for i in range(0, arg.idx * -1):
#                     tmp = Arg(
#                         arg.var,
#                         arg.dim,
#                         arg.typ,
#                         arg.acc,
#                         arg.loc,
#                         arg.map,
#                         i,
#                         arg.soa,
#                         True,
#                     )
#                     tmp.opt = arg.opt
#                     tmp.unique_i = arg.i
#                     tmp.indI = arg.indI
#                     tmp.vec_size = abs(arg.idx)
#                     if i == 0:
#                         tmp.unique = True
#                     else:
#                         tmp.unique = False
#                     self.expanded_args.append(tmp)
#             else:
#                 arg.unique_i = arg.i
#                 arg.unique = True
#                 self.expanded_args.append(arg)
#
#         # Number of arguments after expanding vec args
#         self.nargs = len(self.expanded_args)
#
#         # Assign index of first arg to use this combination of map + idx
#         # Also assign a loop argument index to each arg
#         # Assign index of first arg to use this map
#         # Assign index of first arg to use this dat
#         for i, arg in enumerate(self.expanded_args):
#             arg.i = i
#             arg.datInd = i
#             arg.mapInd = i
#             arg.mapIdxInd = i
#             arg.mapIdxIndFirst = True
#             for j in range(0, i):
#                 if arg.var == self.expanded_args[j].var:
#                     arg.datInd = self.expanded_args[j].datInd
#                 if arg.indirect and (arg.map == self.expanded_args[j].map):
#                     arg.mapInd = self.expanded_args[j].mapInd
#                 if arg.indirect and (arg.map == self.expanded_args[j].map) and (arg.idx == self.expanded_args[j].idx):
#                     arg.mapIdxInd = self.expanded_args[j].mapIdxInd
#                     arg.mapIdxIndFirst = False
#
#         # Calculate cumulative indirect index of OP_INC args
#         i = 0
#         for arg in self.indirects:
#             if arg.acc == "OP_INC":
#                 arg.cumulative_ind_idx = i
#                 i = i + 1
#
#         # Number of unique mappings
#         self.nmaps = 0
#         if self.indirection:
#             self.nmaps = max([arg.mapIdxInd for arg in self.expanded_args]) + 1
#
#     # Name of kernel
#     @property
#     def name(self) -> str:
#         return self.kernel
#
#     # Returns true if any arg is accessing a dat indirectly
#     @property
#     def indirection(self) -> bool:
#         return len(self.indirects) > 0
#
#     # Returns true if no arg is accessing a dat indirectly
#     @property
#     def direct(self) -> bool:
#         return not self.indirection
#
#     # Returns true if any arg uses SoA
#     @property
#     def any_soa(self) -> bool:
#         return len(self.soas) > 0
#
#     # Returns true if any direct arg uses SoA
#     @property
#     def direct_soa(self) -> bool:
#         return any(arg.direct and arg.dim > 1 for arg in self.soas)
#
#     # Gets index of first direct arg using SoA
#     @property
#     def direct_soa_idx(self) -> int:
#         if self.direct_soa:
#             for arg in self.expanded_args:
#                 if arg.direct and arg.soa:
#                     return arg.i
#         return -1
#
#     # List of args accessing dats directly
#     @property
#     def directs(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.direct]
#
#     # List of args accessing dats indirectly
#     @property
#     def indirects(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.indirect]
#
#     # List of args with an option flag set
#     @property
#     def opts(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.opt is not None]
#
#     # List of args which are global loop variables
#     @property
#     def globals(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.global_]
#
#     # List of args which are global OP_READ or OP_WRITE loop variables
#     @property
#     def globals_r_w(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.global_ and (arg.acc == "OP_READ" or arg.acc == "OP_WRITE")]
#
#     # List of args which use vec indexing
#     @property
#     def vecs(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.vector]
#
#     # List of args which use SoA
#     @property
#     def soas(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.soa]
#
#     # List of args where only the first occurrence of each dat is included
#     @property
#     def uniqueVars(self) -> List[Arg]:
#         return uniqueBy(self.expanded_args, lambda a: a.var)
#
#     # List of indirect args where only the first occurrence of each dat is included
#     @property
#     def indirectVars(self) -> List[Arg]:
#         return uniqueBy(self.indirects, lambda a: a.var)
#
#     # List of indirect args where only the first occurrence of each map is included
#     @property
#     def indirectMaps(self) -> List[Arg]:
#         return uniqueBy(self.indirects, lambda a: a.map)
#
#     # List of indirect args where only the first occurrence of each (map, mapping id) pair is included
#     @property
#     def indirectIdxs(self) -> List[Arg]:
#         res = []
#         for arg in self.indirects:
#             if arg.idx is not None and arg.idx < 0:
#                 for i in range(0, abs(arg.idx)):
#                     tmpArg = Arg(arg.var, arg.dim, arg.typ, arg.acc, arg.loc, arg.map, i)
#                     tmpArg.i = arg.i
#                     tmpArg.opt = arg.opt
#                     tmpArg.map1st = arg.map1st
#                     tmpArg.arg1st = arg.arg1st
#                     res.append(tmpArg)
#             else:
#                 res.append(arg)
#         return uniqueBy(res, lambda a: (a.map, a.idx))
#
#     # Indexing the dats referenced by indirect args (with -1 for direct args)
#     # Corresponds to the inds array in the original code gen
#     @property
#     def indirectVarInds(self) -> List[int]:
#         descriptor = []
#
#         for arg in self.expanded_args:
#             if arg.indirect:
#                 for i, a in enumerate(self.indirectVars):
#                     if a.var == arg.var:
#                         descriptor.append(i)
#             else:
#                 descriptor.append(-1)
#
#         return descriptor
#
#     # True if any global reductions used in this loop
#     @property
#     def reduction(self) -> bool:
#         return any(arg.acc != "OP_READ" and arg.acc != "OP_WRITE" for arg in self.globals)
#
#     # True if any multi dim (i.e. dimension of the global is > 1) reduction
#     @property
#     def multiDimReduction(self) -> bool:
#         return self.reduction and any(arg.dim > 1 for arg in self.globals)
#
#     # Get the index of the arg which matches the (map, map id) pair
#     def mapIdxLookup(self, map: str, idx: int) -> int:
#         for i, arg in enumerate(self.indirectIdxs):
#             if arg.map == map and arg.idx == idx:
#                 return i
#
#         return -1
#
#     # Get the unique arguments (the ones corresponding to the original args)
#     @property
#     def unique(self) -> List[Arg]:
#         return [arg for arg in self.expanded_args if arg.unique]
#
#     # Function that allows exceptions to be raised during templating
#     def raise_exception(self, text: str) -> None:
#         exit(text + '. Error from processing "' + self.name + '" loop')
