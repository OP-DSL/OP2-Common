from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import ClassVar, List, Optional, Tuple

import cpp
import fortran
import op as OP
import optimisation
from jinja import env
from language import Lang
from optimisation import Opt
from store import Application, Kernel
from util import find, safeFind


class Scheme(object):
    instances: ClassVar[List[Scheme]] = []

    lang: Lang
    opt: Opt
    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __init__(
        self,
        lang: Lang,
        opt: Opt,
        loop_host_template: Path,
        master_kernel_template: Path = None,
    ) -> None:
        if Scheme.find(lang, opt):
            exit("duplicate scheme")

        self.__class__.instances.append(self)

        self.lang = lang
        self.opt = opt
        self.loop_host_template = loop_host_template
        self.master_kernel_template = master_kernel_template

    @classmethod
    def all(cls) -> List[Opt]:
        return cls.instances

    @classmethod
    def find(cls, lang: Lang, opt: Opt) -> Opt:
        return safeFind(cls.all(), lambda s: s.lang == lang and s.opt == opt)

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

    def translateKernel(self, kernel: Kernel, app: Application) -> str:
        raise NotImplementedError(f'no kernel translator registered for the "{self}" scheme')


cseq = Scheme(
    cpp.lang,
    optimisation.seq,
    Path("cpp/seq/loop_host.cpp.j2"),
    Path("cpp/seq/master_kernel.cpp.j2"),
)

comp = Scheme(
    cpp.lang,
    optimisation.omp,
    Path("cpp/omp/loop_host.cpp.j2"),
    Path("cpp/omp/master_kernel.cpp.j2"),
)

ccuda = Scheme(
    cpp.lang,
    optimisation.cuda,
    Path("cpp/cuda_kepler/loop_host.cu.j2"),
    Path("cpp/cuda_kepler/master_kernel.cu.j2"),
)

from cpp.translator.kernels import cuda

ccuda.translateKernel = MethodType(cuda.translateKernel, ccuda)  # type: ignore

fseq = Scheme(
    fortran.lang,
    optimisation.seq,
    Path("fortran/seq/loop_host.F90.j2"),
)

fvec = Scheme(
    fortran.lang,
    optimisation.vec,
    Path("fortran/vec/loop_host.F90.j2"),
)

fomp = Scheme(
    fortran.lang,
    optimisation.omp,
    Path("fortran/omp/loop_host.F90.j2"),
)

fcuda = Scheme(
    fortran.lang,
    optimisation.cuda,
    Path("fortran/cuda/loop_host.CUF.j2"),
)

from fortran.translator.kernels import cuda

fcuda.translateKernel = MethodType(cuda.translateKernel, fcuda)  # type: ignore

from fortran.translator.kernels import vec

fvec.translateKernel = MethodType(vec.translateKernel, fvec)  # type: ignore
