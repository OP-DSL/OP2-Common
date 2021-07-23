# Standard library imports
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, List, Set, Any
from typing_extensions import Protocol
from os.path import basename
from pathlib import Path

# Application imports
from op import OpError
from util import safeFind, flattern, find, uniqueBy
import op as OP
if TYPE_CHECKING:
  from language import Lang


class Location:
  file: str
  line: int
  column: int

  def __init__(self, file: str, line: int, column: int) -> None:
    self.file = file
    self.line = line
    self.column = column


  def __str__(self) -> str:
    return f'{basename(self.file)}/{self.line}:{self.column}'


class ParseError(Exception):
  message: str
  loc: Optional[Location]

  def __init__(self, message: str, loc: Location = None) -> None:
    self.message = message
    self.loc = loc

  def __str__(self) -> str:
    if self.loc:
      return f'{self.loc}: parse error: {self.message}'
    else:
      return f'parse error: {self.message}'


class Program:
  path: Path
  init: Optional[Location]
  exit: bool
  sets: List[OP.Set]
  maps: List[OP.Map]
  datas: List[OP.Data]
  consts: List[OP.Const]
  loops: List[OP.Loop]


  def __init__(self, path: Path) -> None:
    self.path = path
    self.init = None
    self.exit = False
    self.sets = []
    self.maps = []
    self.datas = []
    self.consts = []
    self.loops = []


  def recordInit(self, loc) -> None:
    if self.init:
      exit('multiple calls to op_init')

    self.init = loc


  def recordExit(self) -> None:
    self.exit = True


  def __str__(self) -> str:
    return f"{'init, ' if self.init else ''}{len(self.consts)} constants, {len(self.loops)} loops{', exit' if self.exit else ''}"



class Kernel:
  name: str
  path: Path
  ast: Any # TODO: Update typing
  params: List[Tuple[str, str]]


  def __init__(self, name: str, path: Path, ast: Any, params: List[Tuple[str, str]]):
    self.name = name
    self.path = path
    self.ast = ast
    self.params = params


  @property
  def paramCount(self) -> int:
    return len(self.params)


  def __str__(self) -> str:
    return self.name



class Application:
  programs: List[Program]
  kernels: List[Kernel]


  def __init__(self):
    self.programs = []
    self.kernels = []


  def validate(self, lang: Lang) -> None:
    if not self.hasInit:
      print('warning: no call to op_init found')

    if not self.hasExit:
      print('warning: no call to op_exit found')

    # Collect the pointers of defined sets
    set_ptrs = [ s.ptr for s in self.sets ]

    # Validate data declerations
    for data in self.datas:
      # Validate set
      if data.set not in set_ptrs:
        raise OpError(f'undefined set "{data.set}" referenced in data decleration', data.loc)

      # Validate type
      if data.typ not in lang.types:
        raise OpError(f'unsupported datatype "{data.typ}" for the {lang.name} language', data.loc)

    # Validate map declerations
    for map in self.maps:
      # Validate both sets
      for set_ in (map.from_set, map.to_set):
        if set_ not in set_ptrs:
          raise OpError(f'undefined set "{set_}" referenced in map decleration', map.loc)

    # Validate constant declerations
    for const in self.consts:
      # Search for previous decleration
      prev = safeFind(self.consts, lambda c: c.ptr == const.ptr)

      if prev and const.dim != prev.dim:
        raise ParseError(f'dim mismatch in repeated decleration of "{const.ptr}" const')
      elif prev and const.dim != prev.dim:
        raise ParseError(f'size mismatch in repeated decleration of "{const.ptr}" const')

    # Validate loop calls
    for loop in self.loops:
      kern = safeFind(self.kernels, lambda k: k.name == loop.kernel)
      loop.kernelPath = str(kern.path)
      prev = safeFind(self.loops, lambda l: l.kernel == loop.kernel)
      if prev:
        for i, (arg_a, arg_b) in enumerate(zip(prev.args, loop.args)):
          if arg_a.acc != arg_b.acc:
            raise ParseError(f'varying access types for arg {i} in {loop.kernel} par loops')
          # TODO: Consider more compatability issues

      # Validate loop dataset
      if loop.set not in set_ptrs:
        raise OpError(f'undefined set "{loop.set}" referenced in par loop call', loop.loc)

      # Validate loop args
      for arg in loop.args:
        if not arg.global_:
          # Look for the referenced data
          data_ = safeFind(self.datas, lambda d: d.ptr == arg.var)

          # Validate the data referenced in the arg
          if not data_:
            raise OpError(f'undefined data "{arg.var}" referenced in par loop arg', arg.loc)
          elif arg.typ != data_.typ:
            raise OpError(f'type mismatch of par loop data, expected {data_.typ}', arg.loc)
          elif arg.dim != data_.dim:
            raise OpError(f'dimension mismatch of par loop data, expected {data_.dim}', arg.loc)

          # Validate direct args
          if arg.direct:
            # Validate index
            if arg.idx != -1:
              raise OpError('incompatible index for direct access, expected -1', arg.loc)
            # Check the dataset can be accessed directly
            if data_.set != loop.set:
              raise OpError(f'cannot directly access the "{arg.var}" dataset from the "{loop.set}" loop set', arg.loc)

            # Check that the same dataset has not already been directly accessed
            if safeFind(loop.directs, lambda a: a is not arg and a.var == arg.var):
              raise OpError(f'duplicate direct accesses to the "{arg.var}" dataset in the same par loop', arg.loc)

          # Validate indirect args
          elif arg.indirect:
            # Look for the referenced map decleration
            map_ = safeFind(self.maps, lambda m: m.ptr == arg.map)

            if not map_:
              raise OpError(f'undefined map "{arg.map}" referenced in par loop arg', arg.loc)

            # Check that the mapping maps from the loop set
            if map_.from_set != loop.set:
              raise OpError(f'cannot apply the "{arg.map}" mapping to the "{loop.set}" loop set', arg.loc)

            # Check that the mapping maps to the data set
            if map_.to_set != data_.set:
              raise OpError(f'cannot map to the "{arg.var}" dataset with the "{arg.map}" mapping', arg.loc)

            # Determine the valid index range using the given language
            min_idx = 0 if lang.zero_idx else 1
            max_idx = map_.dim - 1 if lang.zero_idx else map_.dim

            # Perform range check
            if arg.idx is None or arg.idx < min_idx or arg.idx > max_idx:
              raise OpError(f'index {arg.idx} out of range, must be in the interval [{min_idx},{max_idx}]', arg.loc)

          # Enforce unique data access
          for other in loop.args:
            if other is not arg and other.var == arg.var and (other.idx == arg.idx and other.map == arg.map):
              raise OpError(f'duplicate data accesses in the same par loop', arg.loc)


      # Validate par loop arguments against kernel parameters
      kernel = find(self.kernels, lambda k: k.name == loop.kernel)

      if len(loop.args) != kernel.paramCount:
        raise ParseError(f'incorrect number of args passed to the {kernel} kernel', loop.loc)

      for i, (param, arg) in enumerate(zip(kernel.params, loop.args)):
        if arg.typ != param[1]:
          raise ParseError(f'argument {i} to {kernel} kernel has incompatible type {arg.typ}, expected {param[1]}', arg.loc)



  @property
  def hasInit(self) -> bool:
    return any( program.init for program in self.programs )


  @property
  def hasExit(self) -> bool:
    return any( program.exit for program in self.programs )


  @property
  def sets(self) -> List[OP.Set]:
    return flattern(program.sets for program in self.programs)


  @property
  def maps(self) -> List[OP.Map]:
    return flattern(program.maps for program in self.programs)


  @property
  def datas(self) -> List[OP.Data]:
    return flattern(program.datas for program in self.programs)


  @property
  def consts(self) -> List[OP.Const]:
    consts = flattern(program.consts for program in self.programs)
    return uniqueBy(consts, lambda c: c.ptr)


  @property
  def loops(self) -> List[OP.Loop]:
    loops = flattern(program.loops for program in self.programs)
    return uniqueBy(loops, lambda l: l.kernel)
