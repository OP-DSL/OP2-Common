from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, FrozenSet, List, Optional, Set

from op import Type
from store import Program
from util import Findable


class Lang(Findable):
    name: str

    source_exts: List[str]
    include_ext: str
    kernel_dir: bool

    com_delim: str
    zero_idx: bool

    @abstractmethod
    def parseFile(self, path: Path, include_dirs: FrozenSet[Path], defines: FrozenSet[str]) -> Any:
        pass

    @abstractmethod
    def parseProgram(self, path: Path, include_dirs: Set[Path], defines: List[str]) -> Program:
        pass

    @abstractmethod
    def translateProgram(self, program: Program, include_dirs: Set[Path], defines: List[str], force_soa: bool) -> str:
        pass

    @abstractmethod
    def formatType(self, typ: Type) -> str:
        pass

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name if type(other) is type(self) else False

    def __hash__(self) -> int:
        return hash(self.name)

    def matches(self, key: str) -> bool:
        return key in self.source_exts
