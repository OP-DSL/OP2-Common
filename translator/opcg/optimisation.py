# Standard library imports
from __future__ import annotations
from typing import List, ClassVar, Dict, Any

# Application imports
from util import find


class Opt(object):
  instances: ClassVar[List[Opt]] = []
  name: str
  kernel_translation: bool
  config: Dict[str, Any]

  def __init__(self, name: str, kernel_translation: bool = False, config: Dict[str, Any] = {}) -> None:
    self.__class__.instances.append(self)
    self.name = name
    self.kernel_translation = kernel_translation
    self.config = config


  def __str__(self) -> str:
    return self.name


  def __eq__(self, other) -> bool:
    return self.name == other.name if type(other) is type(self) else False


  def __hash__(self) -> int:
    return hash(self.name)


  @classmethod
  def all(cls) -> List[Opt]:
    return cls.instances


  @classmethod
  def names(cls) -> List[str]:
    return [ o.name for o in cls.all() ]


  @classmethod
  def find(cls, name: str) -> Opt:
    return find(cls.all(), lambda o: o.name == name)


# Define optimisations here ...

seq = Opt('seq', False, config={
  'grouped': False
})

cuda = Opt('cuda', True, config={
  'atomics': True,
  'ind_inc': False,
  'inc_stage': 0,
  'soa': False
})

omp = Opt('omp', False, config={
  'grouped': False
})

vec = Opt('vec', True, config={
  'grouped': False
})
