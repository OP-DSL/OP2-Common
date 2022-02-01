from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import ClassVar, List, Optional, Tuple

import cpp
import cpp.translator.kernels.cuda
import fortran
import fortran.translator.kernels.cuda
import fortran.translator.kernels.vec
import op as OP
import optimisation
from jinja import env
from language import Lang
from optimisation import Opt
from store import Application, Kernel
from util import Findable


class Scheme(Findable):
    lang: Lang
    opt: Opt

    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return self.lang.name + "-" + self.opt.name

    def genLoopHost(self, loop: OP.Loop, i: int) -> Tuple[str, str]:
        # Load the loop host template
        template = env.get_template(str(self.loop_host_template))
        extension = self.loop_host_template.suffixes[-2][1:]

        # Generate source from the template
        return template.render(OP=OP, parloop=loop, opt=self.opt, id=i), extension

    def genMasterKernel(self, app) -> Tuple[str, str]:
        if self.master_kernel_template is None:
            exit(f"No master kernel template registered for {self}")
        # Load the loop host template
        template = env.get_template(str(self.master_kernel_template))
        extension = self.master_kernel_template.suffixes[-2][1:]

        # Generate source from the template
        return template.render(OP=OP, app=app, opt=self.opt), extension

    def translateKernel(self, source: str, kernel: Kernel, app: Application) -> str:
        raise NotImplementedError(f'no kernel translator registered for the "{self}" scheme')

    def matches(self, key: tuple[Lang, Opt]) -> bool:
        return self.lang == key[0] and self.opt == key[1]


class CppSeq(Scheme):
    lang = Lang.find("cpp")
    opt = Opt.find("seq")

    loop_host_template = Path("cpp/seq/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/seq/master_kernel.cpp.j2")


class CppOpenMP(Scheme):
    lang = Lang.find("cpp")
    opt = Opt.find("openmp")

    loop_host_template = Path("cpp/omp/loop_host.cpp.j2")
    master_kernel_template = Path("cpp/omp/master_kernel.cpp.j2")


class CppCuda(Scheme):
    lang = Lang.find("cpp")
    opt = Opt.find("cuda")

    loop_host_template = Path("cpp/cuda_kepler/loop_host.cu.j2")
    master_kernel_template = Path("cpp/cuda_kepler/master_kernel.cu.j2")

    def translateKernel(self, source: str, kernel: Kernel, app: Application) -> str:
        return cpp.translator.kernels.cuda.translateKernel(self.opt.config, source, kernel, app)


Scheme.register(CppSeq)
Scheme.register(CppOpenMP)
Scheme.register(CppCuda)


class FortranSeq(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("seq")

    loop_host_template = Path("fortran/seq/loop_host.F90.j2")
    master_kernel_template = None


class FortranVec(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("vec")

    loop_host_template = Path("fortran/vec/loop_host.F90.j2")
    master_kernel_template = None

    def translateKernel(self, source:str, kernel: Kernel, app: Application) -> str:
        return fortran.translator.kernels.vec.translateKernel(self.opt.config, source, kernel, app)


class FortranOpenMP(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("openmp")

    loop_host_template = Path("fortran/omp/loop_host.F90.j2")
    master_kernel_template = None


class FortranCuda(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("cuda")

    loop_host_template = Path("fortran/cuda/loop_host.CUF.j2")
    master_kernel_template = None

    def translateKernel(self, source: str, kernel: Kernel, app: Application) -> str:
        return fortran.translator.kernels.cuda.translateKernel(self.opt.config, source, kernel, app)


Scheme.register(FortranSeq)
Scheme.register(FortranVec)
Scheme.register(FortranOpenMP)
Scheme.register(FortranCuda)
