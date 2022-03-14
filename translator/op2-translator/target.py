from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from util import Findable


class Target(Findable):
    name: str
    kernel_translation: bool
    config: Dict[str, Any]

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name if type(other) is type(self) else False

    def __hash__(self) -> int:
        return hash(self.name)

    def matches(self, key: str) -> bool:
        return self.name == key.lower()


class Seq(Target):
    name = "seq"
    kernel_translation = False

    config = {"grouped": False, "device": 1}


class Cuda(Target):
    name = "cuda"
    kernel_translation = True

    config = {"grouped": True, "device": 2, "atomics": True, "color2": False}


class OpenMP(Target):
    name = "openmp"
    kernel_translation = False

    config = {"grouped": False, "device": 1, "thread_timing": False}


class Vec(Target):
    name = "vec"
    kernel_translation = True

    config = {"grouped": False, "device": 1}


Target.register(Seq)
Target.register(Cuda)
Target.register(OpenMP)
Target.register(Vec)
