# Standard library imports
from __future__ import annotations
from typing import TYPE_CHECKING, Final, Optional, Dict, List

# Third party imports
from cached_property import cached_property

# Application imports
from util import uniqueBy, find
if TYPE_CHECKING:
  from store import Location


ID: Final[str] = 'OP_ID'

INC: Final[str] = 'OP_INC'
MAX: Final[str] = 'OP_MAX'
MIN: Final[str] = 'OP_MIN'
RW: Final[str] = 'OP_RW'
READ: Final[str] = 'OP_READ'
WRITE: Final[str] = 'OP_WRITE'

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
      return f'{self.loc}: OP error: {self.message}'
    else:
      return f'OP error: {self.message}'


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


  def __init__(
    self,
    from_set: str,
    to_set: str,
    dim: int,
    ptr: str,
    loc: Location
  ) -> None:
    self.from_set = from_set
    self.to_set = to_set
    self.dim = dim
    self.ptr = ptr
    self.loc = loc


class Data:
  set: str
  dim: int
  typ: str
  ptr: str
  loc: Location

  def __init__(
    self,
    set_: str,
    dim: int,
    typ: str,
    ptr: str,
    loc: Location
  ) -> None:
    self.set = set_
    self.dim = dim
    self.typ = typ
    self.ptr = ptr
    self.loc = loc


class Const:
  ptr: str
  dim: int
  typ: str
  debug: str
  loc: Location


  def __init__(self, ptr: str, dim: int, typ: str, debug: str, loc: Location) -> None:
    self.ptr = ptr
    self.dim = dim
    self.typ = typ
    self.debug = debug
    self.loc = loc


class Arg:
  i: int = 0 # Loop argument index
  var: str   # Dataset identifier
  dim: int   # Dataset dimension (redundant)
  typ: str   # Dataset type (redundant)
  acc: str   # Dataset access operation
  loc: Location      # Source code location
  map: Optional[str] # Indirect mapping indentifier
  idx: Optional[int] # Indirect mapping index
  opt: Optional[str]
  map1st: Optional[int] # First arg to use this arg's map
  arg1st: Optional[int] # First arg to use this arg's dat

  def __init__(
    self,
    var: str,
    dim: int,
    typ: str,
    acc: str,
    loc: Location,
    map_: str = None,
    idx: int = None
  ) -> None:
    self.var = var
    self.dim = dim
    self.typ = typ
    self.acc = acc
    self.loc = loc
    self.map = map_
    self.idx = idx
    self.opt = None


  @property
  def direct(self) -> bool:
    return self.map == ID


  @property
  def indirect(self) -> bool:
    return self.map is not None and self.map != ID


  @property
  def global_(self) -> bool:
    return self.map is None


class Loop:
  kernel: str
  set: str
  loc: Location
  args: List[Arg]
  thread_timing: bool
  kernelPath: str


  def __init__(self, kernel: str, set_: str, loc: Location, args: List[Arg]) -> None:
    self.kernel = kernel
    self.set = set_
    self.loc = loc
    self.args = args
    seenMaps = {}
    seenArgs = {}
    for i, arg in enumerate(args):
      arg.i = i
      if arg.var in seenArgs.keys():
          arg.arg1st = seenArgs[arg.var]
      else:
          seenArgs[arg.var] = arg.i
          arg.arg1st = arg.i

    for arg in self.indirects:
      if arg.map in seenMaps.keys():
          arg.map1st = seenMaps[arg.map]
      else:
          seenMaps[arg.map] = arg.i
          arg.map1st = arg.i


  @property
  def name(self) -> str:
    return self.kernel


  @cached_property
  def indirection(self) -> bool:
    return len(self.indirects) > 0


  @cached_property
  def directs(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.direct ]


  @cached_property
  def indirects(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.indirect ]


  @cached_property
  def globals(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.global_ ]


  @cached_property
  def uniqueVars(self) -> List[Arg]:
    return uniqueBy(self.args, lambda a: a.var)


  @cached_property
  def indirectVars(self) -> List[Arg]:
    return uniqueBy(self.indirects, lambda a: a.var)


  @cached_property
  def indirectMaps(self) -> List[Arg]:
    return uniqueBy(self.indirects, lambda a: a.map)


  @cached_property
  def indirectIdxs(self) -> List[Arg]:
    return uniqueBy(self.indirects, lambda a: (a.map, a.idx))


  @cached_property
  def indirectionDescriptor(self) -> List[int]:
    descriptor = []

    for arg in self.args:
      if arg.indirect:
        for i, a in enumerate(self.indirectVars):
          if a.var == arg.var:
            descriptor.append(i)
      else:
        descriptor.append(-1)

    return descriptor


  @cached_property
  def reduction(self) -> bool:
    return any( arg.acc != READ for arg in self.globals )


  @cached_property
  def multiDimReduction(self) -> bool:
    return self.reduction and any( arg.dim > 1 for arg in self.globals )


  def mapIdxLookup(self, map : str, idx : int) -> int:
      for i, arg in enumerate(self.indirectIdxs):
          if arg.map == map and arg.idx == idx:
              return i

      return -1
