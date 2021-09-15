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
  soa: bool          # Does this arg use SoA
  opt: Optional[str]
  map1st: Optional[int] # First arg to use this arg's map
  arg1st: Optional[int] # First arg to use this arg's dat
  optidx: Optional[int] # Index in list of args with opt set
  cumulative_ind_idx: Optional[ind] # Cumulative index of indirect OP_INC args

  def __init__(
    self,
    var: str,
    dim: int,
    typ: str,
    acc: str,
    loc: Location,
    map_: str = None,
    idx: int = None,
    soa: bool = False
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
  def vec(self) -> bool:
    return self.idx is not None and self.idx < -1


class Loop:
  kernel: str
  set: str
  loc: Location
  args: List[Arg]
  thread_timing: bool
  kernelPath: str
  nargs: int


  def __init__(self, kernel: str, set_: str, loc: Location, args: List[Arg]) -> None:
    self.kernel = kernel
    self.set = set_
    self.loc = loc
    self.args = args
    seenMaps = {}
    seenArgs = {}

    # Assign loop argument index to each arg (accounting for vec args)
    i = 0
    for arg in self.args:
      arg.i = i
      if arg.var in seenArgs.keys():
        arg.arg1st = seenArgs[arg.var]
      else:
        seenArgs[arg.var] = arg.i
        arg.arg1st = arg.i
      if arg.vec:
        i += abs(arg.idx)
      else:
        i += 1

    # Store in each arg, which arg first uses the relevant map (for code gen)
    for arg in self.indirects:
      if arg.map in seenMaps.keys():
        arg.map1st = seenMaps[arg.map]
      else:
        seenMaps[arg.map] = arg.i
        arg.map1st = arg.i

    # Number of args for this loop accounting for vec args
    self.nargs = len(self.args)
    for arg in self.vecs:
      self.nargs += abs(arg.idx) - 1

    # Index args which uses opts (args with the same dat + map combination have same index)
    nopts = 0;
    for arg in self.opts:
      if arg.i == arg.arg1st:
        arg.optidx = nopts
        nopts = nopts + 1
      else:
        arg.optidx = self.args[arg.arg1st].optidx

    # Calculate cumulative indirect index of OP_INC args
    i = 0
    for arg in self.indirects:
      if arg.acc == "OP_INC":
        arg.cumulative_ind_idx = i
        i = i + 1

  @property
  def name(self) -> str:
    return self.kernel

  # Returns true if any arg is accessing a dat indirectly
  @cached_property
  def indirection(self) -> bool:
    return len(self.indirects) > 0

  # Returns true if any arg uses SoA
  @cached_property
  def any_soa(self) -> bool:
    return len(self.soas) > 0

  # Returns true if any direct arg uses SoA
  @cached_property
  def direct_soa(self) -> bool:
    return any(arg.direct for arg in self.soas)

  # Gets idx of first direct arg using SoA
  @cached_property
  def direct_soa_idx(self) -> bool:
    if self.direct_soa:
      for arg in self.args:
        if arg.direct and arg.soa:
          return arg.i
    return -1

  # List of args accessing dats directly
  @cached_property
  def directs(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.direct ]

  # List of args accessing dats indirectly
  @cached_property
  def indirects(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.indirect ]

  # List of args with an option flag set
  @cached_property
  def opts(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.opt is not None ]

  # List of args which are global loop variables
  @cached_property
  def globals(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.global_ ]

  # List of args which are global OP_READ or OP_WRITE loop variables
  @cached_property
  def globals_r_w(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.global_ and (arg.acc == "OP_READ" or arg.acc == "OP_WRITE")]

  # List of args which use vec indexing
  @cached_property
  def vecs(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.vec ]

  # List of args which use SoA
  @cached_property
  def soas(self) -> List[Arg]:
    return [ arg for arg in self.args if arg.soa ]

  # List of args where only the first occurrence of each dat is included
  @cached_property
  def uniqueVars(self) -> List[Arg]:
    return uniqueBy(self.args, lambda a: a.var)

  # List of indirect args where only the first occurrence of each dat is included
  @cached_property
  def indirectVars(self) -> List[Arg]:
    return uniqueBy(self.indirects, lambda a: a.var)

  # List of indirect args where only the first occurrence of each map is included
  @cached_property
  def indirectMaps(self) -> List[Arg]:
    return uniqueBy(self.indirects, lambda a: a.map)

  # List of indirect args where only the first occurrence of each (map, mapping id) pair is included
  @cached_property
  def indirectIdxs(self) -> List[Arg]:
    res = []
    for arg in self.indirects:
      if arg.idx < 0:
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

  # Index of the dat referenced by the arg (with -1 for direct args)
  @cached_property
  def indirectionDescriptor(self) -> List[int]:
    descriptor = []

    for arg in self.args:
      if arg.indirect:
        for i, a in enumerate(self.indirectVars):
          if a.var == arg.var:
            if arg.vec:
              for x in range(0, abs(arg.idx)):
                descriptor.append(i)
            else:
              descriptor.append(i)
      else:
        descriptor.append(-1)

    return descriptor

  # True if any global reductions used in this loop
  @cached_property
  def reduction(self) -> bool:
    return any( arg.acc != READ for arg in self.globals )

  # True if any multi dim (i.e. dimension of the global is > 1) reduction
  @cached_property
  def multiDimReduction(self) -> bool:
    return self.reduction and any( arg.dim > 1 for arg in self.globals )

  # Get the index of the arg which matches the (map, map id) pair
  def mapIdxLookup(self, map : str, idx : int) -> int:
      for i, arg in enumerate(self.indirectIdxs):
          if arg.map == map and arg.idx == idx:
              return i

      return -1
