from __future__ import annotations

from typing import Any, Dict

from util import Findable


class Target(Findable["Target"]):
    name: str

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

    def defaultConfig(self) -> Dict[str, Any]:
        return {"device": 1}

class CSeq(Target):
    name = "c_seq"

    def defaultConfig(self) -> Dict[str, Any]:
        return {"device": 1}


class Cuda(Target):
    name = "cuda"

    def defaultConfig(self) -> Dict[str, Any]:
        return {"device": 2, "atomics": True, "color2": False, "gbl_inc_atomic": False}

class Hip(Target):
    name = "hip"

    def defaultConfig(self) -> Dict[str, Any]:
        return {"device": 2, "atomics": True, "color2": False, "gbl_inc_atomic": False}

class CCuda(Target):
    name = "c_cuda"

    def defaultConfig(self) -> Dict[str, Any]:
        return {
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


class OpenMPLegacy(Target):
    name = "openmp_legacy"

    def defaultConfig(self) -> Dict[str, Any]:
        return {
            "vectorise": {"enable": True, "simd_len": 8, "blacklist": []},
            "device": 1,
        }


Target.register(Seq)
Target.register(CSeq)
Target.register(Cuda)
Target.register(Hip)
Target.register(CCuda)
Target.register(CHip)
Target.register(OpenMP)
Target.register(OpenMPLegacy)
