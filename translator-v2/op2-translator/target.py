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

class CSeq(Target):
    name = "c_seq"
    kernel_translation = True

    def defaultConfig(self) -> Dict[str, Any]:
        return {"grouped": False, "device": 1}


class Cuda(Target):
    name = "cuda"
    kernel_translation = True

    def defaultConfig(self) -> Dict[str, Any]:
        return {"grouped": True, "device": 2, "atomics": True, "color2": False, "gbl_inc_atomic": False}


class CCuda(Target):
    name = "c_cuda"
    kernel_translation = True

    def defaultConfig(self) -> Dict[str, Any]:
        return {
            "grouped": True,
            "device": 2,
            "atomics": True,
            "color2": False,
            "gbl_inc_atomic": False,
            "func_prefix": "__device__",
            "hip": False
        }


class CHip(CCuda):
    name = "c_hip"

    def defaultConfig(self) -> Dict[str, Any]:
        config = super().defaultConfig()
        config["hip"] = True

        return config


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

class Sycl(Target):
    name = "sycl"
    kernel_translation = True

    def defaultConfig(self) -> Dict[str, Any]:
        return {"grouped": True, "device": 2, "atomics": False, "color2": True, "gbl_inc_atomic": False}

Target.register(Seq)
Target.register(CSeq)
Target.register(Cuda)
Target.register(CCuda)
Target.register(CHip)
Target.register(OpenMP)
Target.register(Sycl)