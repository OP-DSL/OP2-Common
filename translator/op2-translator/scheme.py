from __future__ import annotations

import dataclasses
from collections import OrderedDict
from pathlib import Path
from types import MethodType
from typing import ClassVar, List, Optional, Set, Tuple

from jinja2 import Environment

import cpp
import cpp.translator.kernels.cuda
import cpp.translator.kernels.seq
import fortran
import fortran.translator.kernels.cuda
import fortran.translator.kernels.vec
import op as OP
from language import Lang
from store import Application, Kernel
from target import Target
from util import Findable, find, safeFind


class LoopHost:
    kernel: Kernel
    kernel_func: str
    kernel_idx: int

    set_: OP.Set

    args: List[Tuple[OP.Arg, int]]
    args_expanded: List[Tuple[OP.Arg, int]]

    # Used dats and maps to the index of the first arg to reference them
    dats: OrderedDict[OP.Dat, int]
    maps: OrderedDict[OP.Map, int]

    def __init__(self, loop: OP.Loop, kernel_func: str, kernel_idx: int, app: Application, lang: Lang) -> None:
        self.kernel = app.kernels[loop.kernel]
        self.kernel_func = kernel_func
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
        self.args.append((arg, idx))

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

    def findDat(self, dat_ptr: str) -> Optional[Tuple[OP.Dat, int]]:
        return safeFind(self.dats.items(), lambda dat: dat[0].ptr == dat_ptr)

    def findMap(self, map_ptr: str) -> Optional[Tuple[OP.Map, int]]:
        return safeFind(self.maps.items(), lambda map_: map_[0].ptr == map_ptr)

    def optIdx(self, arg: OP.Arg) -> Optional[int]:
        idx = 0
        for arg2, _ in self.args:
            if arg2 == arg:
                break

            if arg2.opt:
                idx += 1

        return idx


class Scheme(Findable):
    lang: Lang
    target: Target

    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return f"{self.lang.name}/{self.target.name}"

    def genLoopHost(
        self, include_dirs: Set[Path], env: Environment, loop: OP.Loop, app: Application, kernel_idx: int
    ) -> Tuple[str, str]:
        # Load the loop host template
        template = env.get_template(str(self.loop_host_template))
        extension = self.loop_host_template.suffixes[-2][1:]

        kernel = app.kernels[loop.kernel]
        kernel_func = self.translateKernel(include_dirs, kernel, app)

        loop_host = LoopHost(loop, kernel_func, kernel_idx, app, self.lang)

        # Generate source from the template
        return template.render(OP=OP, lh=loop_host, target=self.target), extension

    def genMasterKernel(self, env: Environment, app: Application, user_types_file: Optional[Path]) -> Tuple[str, str]:
        if self.master_kernel_template is None:
            exit(f"No master kernel template registered for {self}")

        user_types = None
        if user_types_file is not None:
            user_types = user_types_file.read_text()

        # Load the loop host template
        template = env.get_template(str(self.master_kernel_template))
        extension = self.master_kernel_template.suffixes[-2][1:]

        # Generate source from the template
        return template.render(OP=OP, app=app, target=self.target, user_types=user_types), extension

    def translateKernel(self, include_dirs: Set[Path], kernel: Kernel, app: Application) -> str:
        return kernel.path.read_text()

    def matches(self, key: Tuple[Lang, Target]) -> bool:
        return self.lang == key[0] and self.target == key[1]


class CppSeq(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("seq")

    loop_host_template = Path("cpp/seq/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/seq/master_kernel.cpp.jinja")

    def translateKernel(self, include_dirs: Set[Path], kernel: Kernel, app: Application) -> str:
        return cpp.translator.kernels.seq.translateKernel(self.lang, include_dirs, self.target.config, kernel, app)


class CppOpenMP(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("openmp")

    loop_host_template = Path("cpp/openmp/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/openmp/master_kernel.cpp.jinja")

    def translateKernel(self, include_dirs: Set[Path], kernel: Kernel, app: Application) -> str:
        return cpp.translator.kernels.seq.translateKernel(self.lang, include_dirs, self.target.config, kernel, app)


class CppCuda(Scheme):
    lang = Lang.find("cpp")
    target = Target.find("cuda")

    loop_host_template = Path("cpp/cuda/loop_host.hpp.jinja")
    master_kernel_template = Path("cpp/cuda/master_kernel.cu.jinja")

    def translateKernel(self, include_dirs: Set[Path], kernel: Kernel, app: Application) -> str:
        return cpp.translator.kernels.cuda.translateKernel(self.lang, include_dirs, self.target.config, kernel, app)


Scheme.register(CppSeq)
Scheme.register(CppOpenMP)
Scheme.register(CppCuda)


class FortranSeq(Scheme):
    lang = Lang.find("F95")
    target = Target.find("seq")

    loop_host_template = Path("fortran/seq/loop_host.F95.jinja")
    master_kernel_template = None


class FortranVec(Scheme):
    lang = Lang.find("F95")
    target = Target.find("vec")

    loop_host_template = Path("fortran/vec/loop_host.F90.jinja")
    master_kernel_template = None

    def translateKernel(self, include_dirs: Set[Path], kernel: Kernel, app: Application) -> str:
        return fortran.translator.kernels.vec.translateKernel(self.target.config, kernel.path.read_text(), kernel, app)


class FortranOpenMP(Scheme):
    lang = Lang.find("F95")
    target = Target.find("openmp")

    loop_host_template = Path("fortran/omp/loop_host.F90.jinja")
    master_kernel_template = None


class FortranCuda(Scheme):
    lang = Lang.find("F95")
    target = Target.find("cuda")

    loop_host_template = Path("fortran/cuda/loop_host.CUF.jinja")
    master_kernel_template = None

    def translateKernel(self, include_dirs: Set[Path], kernel: Kernel, app: Application) -> str:
        return fortran.translator.kernels.cuda.translateKernel(self.target.config, kernel.path.read_text(), kernel, app)


Scheme.register(FortranSeq)
# Scheme.register(FortranVec)
# Scheme.register(FortranOpenMP)
# Scheme.register(FortranCuda)
