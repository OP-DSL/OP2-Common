from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, List, Optional, Set

from op import Type
from store import Kernel, ParseError, Program
from util import Findable


class Lang(Findable):
    name: str

    source_exts: List[str]
    include_ext: str
    kernel_dir: bool

    com_delim: str
    zero_idx: bool

    @abstractmethod
    def parseFile(self, path: Path, include_dirs: Set[Path]) -> Any:
        pass

    @abstractmethod
    def parseProgram(self, path: Path, include_dirs: Set[Path]) -> Program:
        pass

    @abstractmethod
    def parseKernel(self, path: Path, name: str, include_dirs: Set[Path]) -> Optional[Kernel]:
        pass

    @abstractmethod
    def translateProgram(self, source: str, program: Program) -> str:
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
