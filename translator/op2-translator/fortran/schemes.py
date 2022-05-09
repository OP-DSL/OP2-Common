from pathlib import Path
from typing import List, Set

import fortran.translator.kernels as ftk
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, Kernel
from target import Target


class FortranSeq(Scheme):
    lang = Lang.find("F95")
    target = Target.find("seq")

    loop_host_template = Path("fortran/seq/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/seq/master_kernel.F95.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast = ftk.findKernel(self.lang, kernel, include_dirs, defines)

        ftk.renameKernel(kernel_ast, lambda name: f"{name}_seq")
        ftk.renameConsts(kernel_ast, app, lambda const: f"op2_const_{const}")

        ftk.insertStrides(kernel_ast, kernel, app, lambda dat_ptr: f"op2_dat_{dat_ptr}_stride")

        return str(kernel_ast)


Scheme.register(FortranSeq)


class FortranOpenMP(Scheme):
    lang = Lang.find("F95")
    target = Target.find("openmp")

    loop_host_template = Path("fortran/openmp/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/openmp/master_kernel.F95.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast = ftk.findKernel(self.lang, kernel, include_dirs, defines)

        ftk.renameKernel(kernel_ast, lambda name: f"{name}_seq")
        ftk.renameConsts(kernel_ast, app, lambda const: f"op2_const_{const}")

        ftk.insertStrides(kernel_ast, kernel, app, lambda dat_ptr: f"op2_dat_{dat_ptr}_stride")

        return str(kernel_ast)


Scheme.register(FortranOpenMP)


class FortranCuda(Scheme):
    lang = Lang.find("F95")
    target = Target.find("cuda")

    loop_host_template = Path("fortran/cuda/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/cuda/master_kernel.CUF.jinja")

    def translateKernel(self, include_dirs: Set[Path], defines: List[str], kernel: Kernel, app: Application) -> str:
        kernel_ast = ftk.findKernel(self.lang, kernel, include_dirs, defines)

        ftk.renameKernel(kernel_ast, lambda name: f"{name}_gpu")
        ftk.renameConsts(kernel_ast, app, lambda const: f"op2_const_{const}_d")

        ftk.insertStrides(
            kernel_ast,
            kernel,
            app,
            lambda dat_ptr: f"op2_dat_{dat_ptr}_stride_d",
            skip=lambda arg: arg.access_type == OP.AccessType.INC and self.target.config["atomics"],
        )

        return str(kernel_ast)


Scheme.register(FortranCuda)
