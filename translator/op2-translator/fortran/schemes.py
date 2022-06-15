from pathlib import Path
from typing import List, Set

import fortran.translator.kernels as ftk
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, Kernel
from target import Target
from util import find


class FortranSeq(Scheme):
    lang = Lang.find("F95")
    target = Target.find("seq")

    loop_host_template = Path("fortran/seq/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/seq/master_kernel.F95.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast = ftk.findKernel(self.lang, kernel, include_dirs, defines)

        ftk.renameKernel(kernel_ast, lambda name: f"{name}_seq")
        ftk.renameConsts(kernel_ast, app, lambda const: f"op2_const_{const}")

        return str(kernel_ast)


Scheme.register(FortranSeq)


class FortranOpenMP(Scheme):
    lang = Lang.find("F95")
    target = Target.find("openmp")

    loop_host_template = Path("fortran/openmp/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/openmp/master_kernel.F95.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast = ftk.findKernel(self.lang, kernel, include_dirs, defines)

        ftk.renameKernel(kernel_ast, lambda name: f"{name}_openmp")
        ftk.renameConsts(kernel_ast, app, lambda const: f"op2_const_{const}")

        if not self.target.config["vectorise"]:
            return str(kernel_ast)

        kernel_ast2 = ftk.findKernel(self.lang, kernel, include_dirs, defines)

        ftk.renameKernel(kernel_ast2, lambda name: f"{name}_openmp_simd")
        ftk.renameConsts(kernel_ast2, app, lambda const: f"op2_const_{const}")

        def match_indirect(arg):
            return isinstance(arg, OP.ArgDat) and arg.map_ptr is not None

        def match_gbl_reduction(arg):
            return isinstance(arg, OP.ArgGbl) and arg.access_type in [
                OP.AccessType.INC,
                OP.AccessType.MIN,
                OP.AccessType.MAX,
            ]

        ftk.insertStrides(
            kernel_ast2,
            kernel,
            app,
            lambda arg: "SIMD_LEN",
            match=lambda arg: match_indirect(arg) or match_gbl_reduction(arg),
        )

        return str(kernel_ast) + "\n\n" + str(kernel_ast2)


# Scheme.register(FortranOpenMP)


class FortranCuda(Scheme):
    lang = Lang.find("F95")
    target = Target.find("cuda")

    loop_host_template = Path("fortran/cuda/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/cuda/master_kernel.CUF.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast = ftk.findKernel(self.lang, kernel, include_dirs, defines)

        ftk.renameKernel(kernel_ast, lambda name: f"{name}_gpu")
        ftk.renameConsts(kernel_ast, app, lambda const: f"op2_const_{const}_d")

        def match_soa(arg):
            return isinstance(arg, OP.ArgDat) and find(app.dats(), lambda dat: arg.dat_ptr == dat.ptr).soa

        def skip_atomic_inc(arg):
            return arg.access_type == OP.AccessType.INC and self.target.config["atomics"]

        ftk.insertStrides(
            kernel_ast,
            kernel,
            app,
            lambda arg: f"op2_dat_{arg.dat_ptr}_stride_d",
            match=lambda arg: match_soa(arg) and not skip_atomic_inc(arg),
        )

        return str(kernel_ast)


# Scheme.register(FortranCuda)
