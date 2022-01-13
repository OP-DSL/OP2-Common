# Standard library imports
from __future__ import annotations
from typing import List, ClassVar, Tuple
from types import MethodType
from pathlib import Path

# Application imports
from util import find, safeFind
from store import Kernel, Application
from optimisation import Opt
from language import Lang
from jinja import env
import optimisation
import op as OP
import fortran
import cpp


# A scheme is ...
class Scheme(object):
    instances: ClassVar[List[Scheme]] = []

    def __init__(
        self,
        lang: Lang,
        opt: Opt,
        loop_host_template: Path,
        make_stub_template: Path = None,
        master_kernel_template: Path = None,
    ) -> None:
        if Scheme.find(lang, opt):
            exit("duplicate scheme")

        self.__class__.instances.append(self)
        self.lang = lang
        self.opt = opt
        self.loop_host_template = loop_host_template
        self.make_stub_template = make_stub_template
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
        return template.render(parloop=loop, opt=self.opt, id=i), extension

    def genMasterKernel(self, app) -> Tuple[str, str]:
        if self.master_kernel_template is None:
            exit(f"No master kernel template registered for {self}")
        # Load the loop host template
        template = env.get_template(str(self.master_kernel_template))
        extension = self.master_kernel_template.suffixes[-2][1:]

        # Generate source from the template
        return template.render(app=app, opt=self.opt), extension

    def genMakeStub(self, paths: List[Path]) -> str:
        if self.make_stub_template is None:
            exit(f"No Make stub template registered for {self}")

        # Load the make stub template
        template = env.get_template(str(self.make_stub_template))

        source_paths = [path.name for path in paths]
        object_paths = [path.with_suffix(".o") for path in paths]

        return template.render(
            source_paths=source_paths,
            object_paths=object_paths,
            opt=self.opt,
        )

    def translateKernel(self, kernel: Kernel, app: Application) -> str:
        raise NotImplementedError(f'no kernel translator registered for the "{self}" scheme')


# Register schemes here ...

cseq = Scheme(
    cpp.lang,
    optimisation.seq,
    Path("cpp/seq/loop_host.cpp.j2"),
    None,
    Path("cpp/seq/master_kernel.cpp.j2"),
)
comp = Scheme(
    cpp.lang,
    optimisation.omp,
    Path("cpp/omp/loop_host.cpp.j2"),
    None,
    Path("cpp/omp/master_kernel.cpp.j2"),
)
ccuda = Scheme(
    cpp.lang,
    optimisation.cuda,
    Path("cpp/cuda_kepler/loop_host.cu.j2"),
    None,
    Path("cpp/cuda_kepler/master_kernel.cu.j2"),
)

from cpp.translator.kernels import cuda

ccuda.translateKernel = MethodType(cuda.translateKernel, ccuda)  # type: ignore

fseq = Scheme(
    fortran.lang,
    optimisation.seq,
    Path("fortran/seq/loop_host.F90.j2"),
    Path("fortran/seq/make_stub.make.j2"),
)
fvec = Scheme(
    fortran.lang,
    optimisation.vec,
    Path("fortran/vec/loop_host.F90.j2"),
    Path("fortran/vec/make_stub.make.j2"),
)
fomp = Scheme(
    fortran.lang,
    optimisation.omp,
    Path("fortran/omp/loop_host.F90.j2"),
    Path("fortran/omp/make_stub.make.j2"),
)
fcuda = Scheme(
    fortran.lang,
    optimisation.cuda,
    Path("fortran/cuda/loop_host.CUF.j2"),
    Path("fortran/cuda/make_stub.make.j2"),
)

from fortran.translator.kernels import cuda

fcuda.translateKernel = MethodType(cuda.translateKernel, fcuda)  # type: ignore

from fortran.translator.kernels import vec

fvec.translateKernel = MethodType(vec.translateKernel, fvec)  # type: ignore
