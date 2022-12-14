import copy
from pathlib import Path

import fortran.translator.kernels as ftk
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from util import find


class FortranSeq(Scheme):
    lang = Lang.find("F90")
    target = Target.find("seq")

    loop_host_template = Path("fortran/seq/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/seq/master_kernel.F90.jinja")

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])  # TODO: Loop scope
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        dependencies = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

        if self.lang.user_consts_module is None:
            ftk.renameConsts(kernel_entities + dependencies, app, lambda const: f"op2_const_{const}")

        return ftk.writeSource(kernel_entities + dependencies)


Scheme.register(FortranSeq)


class FortranOpenMP(Scheme):
    lang = Lang.find("F90")
    target = Target.find("openmp")

    loop_host_template = Path("fortran/openmp/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/openmp/master_kernel.F90.jinja")

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])  # TODO: Loop scope
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        dependencies = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

        ftk.renameConsts(kernel_entities + dependencies, app, lambda const: f"op2_const_{const}")

        if not self.target.config["vectorise"]:
            return ftk.writeSource(kernel_entities + dependencies)

        simd_kernel_entities = copy.deepcopy(kernel_entities)
        ftk.renameEntities(simd_kernel_entities, lambda name: f"{name}_simd")

        def match_indirect(arg):
            return isinstance(arg, OP.ArgDat) and arg.map_id is not None

        def match_gbl_reduction(arg):
            return isinstance(arg, OP.ArgGbl) and arg.access_type in [
                OP.AccessType.INC,
                OP.AccessType.MIN,
                OP.AccessType.MAX,
            ]

        for simd_kernel_entity in simd_kernel_entities:
            ftk.insertStrides(
                simd_kernel_entity,
                loop,
                app,
                lambda arg: "SIMD_LEN",
                match=lambda arg: match_indirect(arg) or match_gbl_reduction(arg),
            )

        return ftk.writeSource(kernel_entities + simd_kernel_entities + dependencies)


# Scheme.register(FortranOpenMP)


class FortranCuda(Scheme):
    lang = Lang.find("F90")
    target = Target.find("cuda")

    loop_host_template = Path("fortran/cuda/loop_host.inc.jinja")
    master_kernel_template = Path("fortran/cuda/master_kernel.CUF.jinja")

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])  # TODO: Loop scope
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        dependencies = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

        ftk.renameConsts(kernel_entities + dependencies, app, lambda const: f"op2_const_{const}_d")

        def match_soa(arg):
            return isinstance(arg, OP.ArgDat) and loop.dat(arg).soa

        def match_atomic_inc(arg):
            return arg.access_type == OP.AccessType.INC and self.target.config["atomics"]

#       for kernel_entity in kernel_entities:
#           ftk.insertStrides(
#               kernel_entity,
#               loop,
#               app,
#               lambda arg: f"op2_dat{arg.dat_id}_stride_d",
#               match=lambda arg: match_soa(arg) and not match_atomic_inc(arg),
#           )

        return ftk.writeSource(kernel_entities + dependencies, "attributes(device) &\n")


Scheme.register(FortranCuda)
