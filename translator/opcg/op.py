from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Final, List, Optional, Union, Callable

from cached_property import cached_property

from util import find, uniqueBy

if TYPE_CHECKING:
    from store import Location


ID: Final[str] = "OP_ID"

INC: Final[str] = "OP_INC"
MAX: Final[str] = "OP_MAX"
MIN: Final[str] = "OP_MIN"
RW: Final[str] = "OP_RW"
READ: Final[str] = "OP_READ"
WRITE: Final[str] = "OP_WRITE"

DAT_ACCESS_TYPES = [READ, WRITE, RW, INC]
GBL_ACCESS_TYPES = [READ, INC, MAX, MIN]


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


@dataclass
class Float(Type):
    size: int


@dataclass
class Bool(Type):
    pass


class Set:
    ptr: str

    def __init__(self, ptr: str) -> None:
        self.ptr = ptr


class Map:
    from_set: str
    to_set: str
    dim: int
    ptr: str
    loc: Location

    def __init__(self, from_set: str, to_set: str, dim: int, ptr: str, loc: Location) -> None:
        self.from_set = from_set
        self.to_set = to_set
        self.dim = dim
        self.ptr = ptr
        self.loc = loc


class Data:
    set: str
    dim: int
    typ: Type
    ptr: str
    loc: Location

    def __init__(self, set_: str, dim: int, typ: Type, ptr: str, loc: Location) -> None:
        self.set = set_
        self.dim = dim
        self.typ = typ
        self.ptr = ptr
        self.loc = loc


class Const:
    ptr: str
    dim: int
    typ: Type
    debug: str
    loc: Location

    def __init__(self, ptr: str, dim: int, typ: Type, debug: str, loc: Location) -> None:
        self.ptr = ptr
        self.dim = dim
        self.typ = typ
        self.debug = debug
        self.loc = loc


class Arg:
    i: int = 0  # Loop argument index
    indI: int = 0  # Loop argument index for indirect args
    var: str  # Dataset identifier
    dim: int  # Dataset dimension (redundant)
    typ: Type  # Dataset type (redundant)
    acc: str  # Dataset access operation
    loc: Location  # Source code location
    map: Optional[str]  # Indirect mapping indentifier
    idx: Optional[int]  # Indirect mapping index
    soa: bool  # Does this arg use SoA
    opt: Optional[str]
    vec: bool  # Was this arg a vector arg before being expanded
    vec_size: int  # The size of the vector arg before it was expanded
    unique_i: int  # the loop argument index of the unique arg (i.e. arg before vec was expanded)
    unique: bool  # Does this arg represent the arg pre expansion
    datInd: int  # First arg to use this arg's dat
    mapInd: int  # First arg to use this arg's map
    mapIdxInd: int  # First arg to use this arg's map + idx combination
    mapIdxIndFirst: bool  # Is this the first arg to use this arg's map + idx combination
    map1st: Optional[int]  # First arg to use this arg's map
    arg1st: Optional[int]  # First arg to use this arg's dat
    optidx: Optional[int]  # Index in list of args with opt set
    cumulative_ind_idx: Optional[int]  # Cumulative index of indirect OP_INC args

    def __init__(
        self,
        var: str,
        dim: int,
        typ: Type,
        acc: str,
        loc: Location,
        map_: str = None,
        idx: int = None,
        soa: bool = False,
        vec: bool = False,
    ) -> None:
        self.var = var
        self.dim = dim
        self.typ = typ
        self.acc = acc
        self.loc = loc
        self.map = map_
        self.idx = idx
        self.soa = soa
        self.opt = None
        self.vec = vec

    @property
    def direct(self) -> bool:
        return self.map == ID

    @property
    def indirect(self) -> bool:
        return self.map is not None and self.map != ID

    @property
    def global_(self) -> bool:
        return self.map is None

    @property
    def vector(self) -> bool:
        return self.idx is not None and self.idx < 0 and self.indirect


class Loop:
    loc: Location
    kernel: str
    kernelPath: str

    set: str
    args: List[Arg]
    expanded_args: List[Arg]

    nargs: int
    ninds: int

    def __init__(self, kernel: str, set_: str, loc: Location, args: List[Arg]) -> None:
        self.kernel = kernel
        self.set = set_
        self.loc = loc
        self.args = args
        self.expanded_args = []

        ind_i = 0
        for i, arg in enumerate(self.args):
            arg.i = i
            if arg.indirect:
                arg.indI = ind_i
                ind_i += 1
        self.ninds = ind_i

        # Number of arguments for this loop if no vec arguments
        self.nargs_novec = len(self.args)

        # Create a new list of args by expanding the vec args
        for arg in self.args:
            if arg.idx is not None and arg.idx < 0 and arg.indirect:
                # Expand vec arg into multiple args
                for i in range(0, arg.idx * -1):
                    tmp = Arg(
                        arg.var,
                        arg.dim,
                        arg.typ,
                        arg.acc,
                        arg.loc,
                        arg.map,
                        i,
                        arg.soa,
                        True,
                    )
                    tmp.opt = arg.opt
                    tmp.unique_i = arg.i
                    tmp.indI = arg.indI
                    tmp.vec_size = abs(arg.idx)
                    if i == 0:
                        tmp.unique = True
                    else:
                        tmp.unique = False
                    self.expanded_args.append(tmp)
            else:
                arg.unique_i = arg.i
                arg.unique = True
                self.expanded_args.append(arg)

        # Number of arguments after expanding vec args
        self.nargs = len(self.expanded_args)

        # Assign index of first arg to use this combination of map + idx
        # Also assign a loop argument index to each arg
        # Assign index of first arg to use this map
        # Assign index of first arg to use this dat
        for i, arg in enumerate(self.expanded_args):
            arg.i = i
            arg.datInd = i
            arg.mapInd = i
            arg.mapIdxInd = i
            arg.mapIdxIndFirst = True
            for j in range(0, i):
                if arg.var == self.expanded_args[j].var:
                    arg.datInd = self.expanded_args[j].datInd
                if arg.indirect and (arg.map == self.expanded_args[j].map):
                    arg.mapInd = self.expanded_args[j].mapInd
                if arg.indirect and (arg.map == self.expanded_args[j].map) and (arg.idx == self.expanded_args[j].idx):
                    arg.mapIdxInd = self.expanded_args[j].mapIdxInd
                    arg.mapIdxIndFirst = False

        # Calculate cumulative indirect index of OP_INC args
        i = 0
        for arg in self.indirects:
            if arg.acc == "OP_INC":
                arg.cumulative_ind_idx = i
                i = i + 1

        # Number of unique mappings
        self.nmaps = 0
        if self.indirection:
            self.nmaps = max([arg.mapIdxInd for arg in self.expanded_args]) + 1

        ### OLD STUFF (Remove once I've checked its no longer needed) ###
        # seenMaps = {}
        # seenArgs = {}

        # # Assign loop argument index to each arg (accounting for vec args)
        # i = 0
        # for arg in self.args:
        #   arg.i = i
        #   if arg.var in seenArgs.keys():
        #     arg.arg1st = seenArgs[arg.var]
        #   else:
        #     seenArgs[arg.var] = arg.i
        #     arg.arg1st = arg.i
        #   if arg.vector:
        #     i += abs(arg.idx)
        #   else:
        #     i += 1

        # # Store in each arg, which arg first uses the relevant map (for code gen)
        # for arg in self.indirects:
        #   if arg.map in seenMaps.keys():
        #     arg.map1st = seenMaps[arg.map]
        #   else:
        #     seenMaps[arg.map] = arg.i
        #     arg.map1st = arg.i

        # # Index args which uses opts (args with the same dat + map combination have same index)
        # nopts = 0;
        # for arg in self.opts:
        #   if arg.i == arg.arg1st:
        #     arg.optidx = nopts
        #     nopts = nopts + 1
        #   else:
        #     arg.optidx = self.args[arg.arg1st].optidx

    # Name of kernel
    @property
    def name(self) -> str:
        return self.kernel

    # Returns true if any arg is accessing a dat indirectly
    @property
    def indirection(self) -> bool:
        return len(self.indirects) > 0

    # Returns true if no arg is accessing a dat indirectly
    @property
    def direct(self) -> bool:
        return not self.indirection

    # Returns true if any arg uses SoA
    @property
    def any_soa(self) -> bool:
        return len(self.soas) > 0

    # Returns true if any direct arg uses SoA
    @property
    def direct_soa(self) -> bool:
        return any(arg.direct and arg.dim > 1 for arg in self.soas)

    # Gets index of first direct arg using SoA
    @property
    def direct_soa_idx(self) -> int:
        if self.direct_soa:
            for arg in self.expanded_args:
                if arg.direct and arg.soa:
                    return arg.i
        return -1

    # List of args accessing dats directly
    @property
    def directs(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.direct]

    # List of args accessing dats indirectly
    @property
    def indirects(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.indirect]

    # List of args with an option flag set
    @property
    def opts(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.opt is not None]

    # List of args which are global loop variables
    @property
    def globals(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.global_]

    # List of args which are global OP_READ or OP_WRITE loop variables
    @property
    def globals_r_w(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.global_ and (arg.acc == "OP_READ" or arg.acc == "OP_WRITE")]

    # List of args which use vec indexing
    @property
    def vecs(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.vector]

    # List of args which use SoA
    @property
    def soas(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.soa]

    # List of args where only the first occurrence of each dat is included
    @property
    def uniqueVars(self) -> List[Arg]:
        return uniqueBy(self.expanded_args, lambda a: a.var)

    # List of indirect args where only the first occurrence of each dat is included
    @property
    def indirectVars(self) -> List[Arg]:
        return uniqueBy(self.indirects, lambda a: a.var)

    # List of indirect args where only the first occurrence of each map is included
    @property
    def indirectMaps(self) -> List[Arg]:
        return uniqueBy(self.indirects, lambda a: a.map)

    # List of indirect args where only the first occurrence of each (map, mapping id) pair is included
    @property
    def indirectIdxs(self) -> List[Arg]:
        res = []
        for arg in self.indirects:
            if arg.idx is not None and arg.idx < 0:
                for i in range(0, abs(arg.idx)):
                    tmpArg = Arg(arg.var, arg.dim, arg.typ, arg.acc, arg.loc, arg.map, i)
                    tmpArg.i = arg.i
                    tmpArg.opt = arg.opt
                    tmpArg.map1st = arg.map1st
                    tmpArg.arg1st = arg.arg1st
                    res.append(tmpArg)
            else:
                res.append(arg)
        return uniqueBy(res, lambda a: (a.map, a.idx))

    # Indexing the dats referenced by indirect args (with -1 for direct args)
    # Corresponds to the inds array in the original code gen
    @property
    def indirectVarInds(self) -> List[int]:
        descriptor = []

        for arg in self.expanded_args:
            if arg.indirect:
                for i, a in enumerate(self.indirectVars):
                    if a.var == arg.var:
                        descriptor.append(i)
            else:
                descriptor.append(-1)

        return descriptor

    # True if any global reductions used in this loop
    @property
    def reduction(self) -> bool:
        return any(arg.acc != "OP_READ" and arg.acc != "OP_WRITE" for arg in self.globals)

    # True if any multi dim (i.e. dimension of the global is > 1) reduction
    @property
    def multiDimReduction(self) -> bool:
        return self.reduction and any(arg.dim > 1 for arg in self.globals)

    # Get the index of the arg which matches the (map, map id) pair
    def mapIdxLookup(self, map: str, idx: int) -> int:
        for i, arg in enumerate(self.indirectIdxs):
            if arg.map == map and arg.idx == idx:
                return i

        return -1

    # Get the unique arguments (the ones corresponding to the original args)
    @property
    def unique(self) -> List[Arg]:
        return [arg for arg in self.expanded_args if arg.unique]

    # Function that allows exceptions to be raised during templating
    def raise_exception(self, text: str) -> None:
        exit(text + '. Error from processing "' + self.name + '" loop')
