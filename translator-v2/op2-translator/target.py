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

class SeqF2C(Target):
    name = "seq_f2c"

    def defaultConfig(self) -> Dict[str, Any]:
        return {"device": 1}


class CudaLegacy(Target):
    name = "cuda_legacy"

    def defaultConfig(self) -> Dict[str, Any]:
        return {"device": 2, "atomics": True, "color2": False, "gbl_inc_atomic": False}

class HipLegacy(Target):
    name = "hip_legacy"

    def defaultConfig(self) -> Dict[str, Any]:
        return {"device": 2, "atomics": True, "color2": False, "gbl_inc_atomic": False}

class Cuda(Target):
    name = "cuda"

    def defaultConfig(self) -> Dict[str, Any]:
        return {
            "gbl_inc_atomic": False,
            "func_prefix": "__device__",
        }


class Hip(Cuda):
    name = "hip"


class CudaF2C(Cuda):
    name = "cuda_f2c"


class HipF2C(Hip):
    name = "hip_f2c"


class OpenMPF2C(Target):
    name = "openmp_f2c"

    def defaultConfig(self) -> Dict[str, Any]:
        # Kernels inlined into each body that calls them.
        return {"func_prefix": "__attribute__((always_inline)) inline"}


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
Target.register(SeqF2C)
Target.register(CudaLegacy)
Target.register(HipLegacy)
Target.register(Cuda)
Target.register(Hip)
Target.register(CudaF2C)
Target.register(HipF2C)
Target.register(OpenMPF2C)
Target.register(OpenMP)
Target.register(OpenMPLegacy)

# The names the targets had before, each with what replaced it.
renamed = {
    "c_seq": "seq_f2c",
    "c_openmp": "openmp_f2c",
    "c_cuda": "cuda_f2c for Fortran, cuda for C++",
    "c_hip": "hip_f2c for Fortran, hip for C++",
}
