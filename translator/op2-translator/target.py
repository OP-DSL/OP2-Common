from __future__ import annotations

from typing import Any, Dict

from util import Findable


class Target(Findable["Target"]):
    name: str
    kernel_translation: bool

    def defaultConfig(self) -> Dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.name == other.name if type(other) is type(self) else False

    def __hash__(self) -> int:
        return hash(self.name)

    def matches(self, key: Any) -> bool:
        return isinstance(key, str) and self.name == key.lower()


class Seq(Target):
    name = "seq"
    kernel_translation = False

    def defaultConfig(self) -> Dict[str, Any]:
        return {"grouped": False, "device": 1}


class Cuda(Target):
    name = "cuda"
    kernel_translation = True

    def defaultConfig(self) -> Dict[str, Any]:
        return {"grouped": True, "device": 2, "atomics": True, "color2": False, "gbl_inc_atomic": False}


class OpenMP(Target):
    name = "openmp"
    kernel_translation = False

    def defaultConfig(self) -> Dict[str, Any]:
        return {
            "grouped": False,
            "vectorise": {"enable": True, "simd_len": 8, "blacklist": []},
            "device": 1,
            "thread_timing": False,
        }


Target.register(Seq)
Target.register(Cuda)
Target.register(OpenMP)
