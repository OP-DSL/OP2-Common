from __future__ import annotations

import dataclasses
from collections import OrderedDict
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
from util import Findable, find


class LoopHost:
    kernel: Kernel
    kernel_func: str
    kernel_idx: int

    set_: OP.Set

    args: List[OP.Arg]
    args_expanded: List[tuple[OP.Arg, int]]

    # Used dats and maps to the index of the first arg to reference them
    dats: OrderedDict[OP.Dat, int]
    maps: OrderedDict[OP.Map, int]

    def __init__(self, loop: OP.Loop, kernel_idx: int, app: Application, lang: Lang) -> None:
        self.kernel = app.kernels[loop.kernel]
        self.kernel_func = self.kernel.path.read_text()
        self.kernel_idx = kernel_idx

        self.set_ = find(app.sets(), lambda s: s.ptr == loop.set_ptr)

        self.args = []
        self.args_expanded = []

        self.dats = OrderedDict()
        self.maps = OrderedDict()

        for arg in loop.args:
            self.addArg(arg, app, lang)

    def addArg(self, arg: OP.Arg, app: Application, lang: Lang) -> None:
        idx = len(self.args)
        self.args.append(arg)

        if isinstance(arg, OP.ArgGbl):
            self.args_expanded.append((arg, idx))
            return

        dat = find(app.dats(), lambda d: d.ptr == arg.dat_ptr)
        if dat not in self.dats:
            self.dats[dat] = idx

        if arg.map_ptr is None:
            self.args_expanded.append((arg, idx))
            return

        map_ = find(app.maps(), lambda m: m.ptr == arg.map_ptr)
        if map_ not in self.maps:
            self.maps[map_] = idx

        if arg.map_idx >= 0:
            self.args_expanded.append((arg, idx))
            return

        for map_idx in range(-arg.map_idx):
            arg_expanded = dataclasses.replace(arg, map_idx=map_idx)

            if not lang.zero_idx:
                arg_expanded.map_idx += 1

            self.args_expanded.append((arg_expanded, idx))


class Scheme(Findable):
    lang: Lang
    opt: Opt

    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return self.lang.name + "/" + self.opt.name

    def genLoopHost(self, loop: OP.Loop, app: Application, kernel_idx: int) -> Tuple[str, str]:
        # Load the loop host template
        template = env.get_template(str(self.loop_host_template))
        extension = self.loop_host_template.suffixes[-2][1:]

        loop_host = LoopHost(loop, kernel_idx, app, self.lang)

        # Generate source from the template
        return template.render(OP=OP, lh=loop_host, opt=self.opt), extension

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

    loop_host_template = Path("cpp/seq/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/seq/master_kernel.cpp.jinja")


class CppOpenMP(Scheme):
    lang = Lang.find("cpp")
    opt = Opt.find("openmp")

    loop_host_template = Path("cpp/omp/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/omp/master_kernel.cpp.jinja")


class CppCuda(Scheme):
    lang = Lang.find("cpp")
    opt = Opt.find("cuda")

    loop_host_template = Path("cpp/cuda/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/cuda/master_kernel.cu.jinja")

    def translateKernel(self, source: str, kernel: Kernel, app: Application) -> str:
        return cpp.translator.kernels.cuda.translateKernel(self.opt.config, source, kernel, app)


Scheme.register(CppSeq)
# Scheme.register(CppOpenMP)
Scheme.register(CppCuda)


class FortranSeq(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("seq")

    loop_host_template = Path("fortran/seq/loop_host.F90.jinja")
    master_kernel_template = None


class FortranVec(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("vec")

    loop_host_template = Path("fortran/vec/loop_host.F90.jinja")
    master_kernel_template = None

    def translateKernel(self, source: str, kernel: Kernel, app: Application) -> str:
        return fortran.translator.kernels.vec.translateKernel(self.opt.config, source, kernel, app)


class FortranOpenMP(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("openmp")

    loop_host_template = Path("fortran/omp/loop_host.F90.jinja")
    master_kernel_template = None


class FortranCuda(Scheme):
    lang = Lang.find("F95")
    opt = Opt.find("cuda")

    loop_host_template = Path("fortran/cuda/loop_host.CUF.jinja")
    master_kernel_template = None

    def translateKernel(self, source: str, kernel: Kernel, app: Application) -> str:
        return fortran.translator.kernels.cuda.translateKernel(self.opt.config, source, kernel, app)


# Scheme.register(FortranSeq)
# Scheme.register(FortranVec)
# Scheme.register(FortranOpenMP)
# Scheme.register(FortranCuda)
