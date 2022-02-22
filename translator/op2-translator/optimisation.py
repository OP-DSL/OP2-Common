from __future__ import annotations

from typing import Any, ClassVar, Dict, List

from util import Findable


class Opt(Findable):
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


class Seq(Opt):
    name = "seq"
    kernel_translation = False

    config = {"grouped": False}


class Cuda(Opt):
    name = "cuda"
    kernel_translation = True

    config = {"atomics": True, "ind_inc": False, "inc_stage": 0}


class OpenMP(Opt):
    name = "openmp"
    kernel_translation = False

    config = {"grouped": False, "thread_timing": False}


class Vec(Opt):
    name = "vec"
    kernel_translation = True

    config = {"grouped": False}


Opt.register(Seq)
Opt.register(Cuda)
Opt.register(OpenMP)
Opt.register(Vec)
