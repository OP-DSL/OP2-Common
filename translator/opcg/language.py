# Standard library imports
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, ClassVar, Set
from pathlib import Path

# Application imports
from store import Program, Kernel, ParseError
from util import find


class Lang(object):
    instances: ClassVar[List[Lang]] = []

    name: str
    com_delim: str
    types: List[str]
    source_exts: List[str]
    include_ext: str
    zero_idx: bool
    kernel_dir: bool

    def __init__(
        self,
        name: str,
        com_delim: str,
        types: List[str],
        source_exts: List[str],
        include_ext: str,
        zero_idx: bool = True,
        kernel_dir: bool = False,
    ) -> None:
        self.__class__.instances.append(self)
        self.name = name
        self.com_delim = com_delim
        self.types = types
        self.source_exts = source_exts
        self.include_ext = include_ext
        self.zero_idx = zero_idx
        self.kernel_dir = kernel_dir

    def parseProgram(self, path: Path, include_dirs: Set[Path]) -> Program:
        raise NotImplementedError(f'no program parser registered for the "{self.name}" language')

    def parseKernel(self, path: Path, name: str) -> Kernel:
        raise NotImplementedError(f'no kernel parser registered for the "{self.name}" language')

    # Augment source program to use generated kernel hosts
    def translateProgram(self, source: str, program: Program, soa: bool = False) -> str:
        raise NotImplementedError(f'no program translator registered for the "{self.name}" language')

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name if type(other) is type(self) else False

    def __hash__(self) -> int:
        return hash(self.name)

    @classmethod
    def all(cls) -> List[Lang]:
        return cls.instances

    @classmethod
    def find(cls, name: str) -> Lang:
        return find(cls.all(), lambda l: name == l.name or name in l.source_exts)
