from __future__ import annotations

from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, FrozenSet, List, Optional, Set, Dict

from op import Type
from store import Application, Program
from util import Findable


class Lang(Findable["Lang"]):
    name: str

    source_exts: List[str]
    include_ext: str
    kernel_dir: bool

    com_delim: str
    ast_is_serializable: bool

    fallback_wrapper_template: Optional[Path]

    @abstractmethod
    def addArgs(self, parser: ArgumentParser) -> None:
        pass

    @abstractmethod
    def parseArgs(self, args: Namespace) -> None:
        pass

    @abstractmethod
    def validate(self, app: Application) -> None:
        pass

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

    def matches(self, key: Any) -> bool:
        return isinstance(key, str) and key in self.source_exts
