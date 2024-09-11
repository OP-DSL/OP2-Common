import copy
import traceback
import sys
from pathlib import Path
from typing import Any, Dict

import fortran.translator.kernels as ftk
import fortran.translator.kernels_c as ftk_c
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from util import find


class FortranSeq(Scheme):
    lang = Lang.get("F90")
    target = Target.get("seq")

    fallback = None

    consts_template = Path("fortran/seq/consts.F90.jinja")
    loop_host_templates = [Path("fortran/seq/loop_host.F90.jinja")]
    master_kernel_templates = [Path("fortran/seq/master_kernel.F90.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])  # TODO: Loop scope
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        dependencies, _ = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

        if self.lang.user_consts_module is None:
            ftk.renameConsts(self.lang, kernel_entities + dependencies, app, lambda const: f"op2_const_{const}")

        return ftk.writeSource(kernel_entities + dependencies)


Scheme.register(FortranSeq)


class FortranOpenMP(Scheme):
    lang = Lang.get("F90")
    target = Target.get("openmp")

    fallback = Scheme.get((Lang.get("F90"), Target.get("seq")))

    consts_template = None
    loop_host_templates = [Path("fortran/openmp/loop_host.inc.jinja")]
    master_kernel_templates = [Path("fortran/openmp/master_kernel.F90.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])  # TODO: Loop scope
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        dependencies, _ = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

        ftk.renameConsts(self.lang, kernel_entities + dependencies, app, lambda const: f"op2_const_{const}")

        if not config["vectorise"]:
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
    lang = Lang.get("F90")
    target = Target.get("cuda")

    fallback = Scheme.get((Lang.get("F90"), Target.get("seq")))

    consts_template = Path("fortran/cuda/consts.F90.jinja")
    loop_host_templates = [Path("fortran/cuda/loop_host.CUF.jinja")]
    master_kernel_templates = [Path("fortran/cuda/master_kernel.F90.jinja")]

    def canGenLoopHost(self, loop: OP.Loop) -> bool:
        for arg in loop.args:
            if isinstance(arg, OP.ArgGbl) and arg.access_type in [OP.AccessType.RW, OP.AccessType.WRITE]:
                return False

        return True

    def getBaseConfig(self, loop: OP.Loop) -> Dict[str, Any]:
        config = self.target.defaultConfig()

        use_coloring = False

        for arg in loop.args:
            if isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.RW:
                use_coloring = True
                break

        if use_coloring:
            config["atomics"] = False
            config["color2"] = True

        return config

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])  # TODO: Loop scope
        if len(kernel_entities) == 0:
            raise ParseError(f"unable to find kernel function: {loop.kernel}")

        if len(kernel_entities) > 1:
            raise ParseError(f"ambiguous kernel function: {loop.kernel}")

        dependencies, _ = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

        ftk.renameConsts(self.lang, kernel_entities + dependencies, app, lambda const: f"op2_const_{const}_d")

        for entity in kernel_entities + dependencies:
            ftk.fixHydraIO(entity)

        for entity in kernel_entities + dependencies:
            ftk.removeExternals(entity)

        def match_indirect(arg):
            return isinstance(arg, OP.ArgDat) and arg.map_id is not None

        def match_soa(arg):
            return isinstance(arg, OP.ArgDat) and loop.dat(arg).soa

        def match_atomic_inc(arg):
            return arg.access_type == OP.AccessType.INC and config["atomics"]

        def match_gbl(arg):
            return isinstance(arg, OP.ArgGbl)

        def match_info(arg):
            return isinstance(arg, OP.ArgInfo)

        def match_reduction(arg):
            return (not config["gbl_inc_atomic"]) and arg.access_type in [OP.AccessType.INC, OP.AccessType.MIN, OP.AccessType.MAX]

        def match_work(arg):
            return arg.access_type == OP.AccessType.WORK

        modified = ftk.insertStrides(
            kernel_entities[0],
            kernel_entities + dependencies,
            loop,
            app,
            lambda arg: f"direct",
            lambda arg: match_soa(arg) and not match_indirect(arg),
        )

        modified = ftk.insertStrides(
            kernel_entities[0],
            kernel_entities + dependencies,
            loop,
            app,
            lambda arg: f"dat{arg.dat_id}",
            lambda arg: match_soa(arg) and match_indirect(arg),
            modified,
        )

        modified = ftk.insertStrides(
            kernel_entities[0],
            kernel_entities + dependencies,
            loop,
            app,
            lambda arg: f"gbl",
            lambda arg: (match_gbl(arg) and (match_reduction(arg) or match_work(arg))) or match_info(arg),
            modified,
        )

        ftk.insertAtomicIncs(
            kernel_entities[0],
            kernel_entities + dependencies,
            loop,
            app,
            lambda arg: match_indirect(arg) and match_atomic_inc(arg),
        )

        if config["gbl_inc_atomic"]:
            ftk.insertAtomicIncs(
                kernel_entities[0],
                kernel_entities + dependencies,
                loop,
                app,
                lambda arg: match_gbl(arg) and arg.access_type == OP.AccessType.INC,
            )

        return ftk.writeSource(kernel_entities + dependencies, "attributes(device) &\n")


Scheme.register(FortranCuda)


class FortranCSeq(Scheme):
    lang = Lang.get("F90")
    target = Target.get("c_seq")

    fallback = Scheme.get((Lang.get("F90"), Target.get("seq")))

    consts_template = None
    loop_host_templates = [Path("fortran/c_seq/loop_host.F90.jinja"), Path("fortran/c_seq/loop_host.cpp.jinja")]
    master_kernel_templates = [Path("fortran/c_seq/master_kernel.F90.jinja")]

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])

        assert(len(kernel_entities) == 1)
        kernel_entity = kernel_entities[0]

        dependencies, _ = ftk.extractDependencies([kernel_entity], app, [])

        kernel_entity = copy.deepcopy(kernel_entity)
        dependencies = copy.deepcopy(dependencies)

        for entity in [kernel_entity] + dependencies:
            ftk.fixHydraIO(entity)

        for entity in [kernel_entity] + dependencies:
            ftk.removeExternals(entity)

        info = ftk_c.parseInfo([kernel_entity] + dependencies, app, loop, config)
        return ftk_c.translate(info)


Scheme.register(FortranCSeq)


class FortranCCuda(Scheme):
    lang = Lang.get("F90")
    target = Target.get("c_cuda")

    fallback = Scheme.get((Lang.get("F90"), Target.get("seq")))

    consts_template = None
    loop_host_templates = [Path("fortran/c_cuda/loop_host.F90.jinja"), Path("fortran/c_cuda/loop_host.cuh.jinja")]
    master_kernel_templates = [Path("fortran/c_cuda/master_kernel.F90.jinja"), Path("fortran/c_cuda/master_kernel.cu.jinja")]

    def canGenLoopHost(self, loop: OP.Loop) -> bool:
        for arg in loop.args:
            if isinstance(arg, OP.ArgGbl) and arg.access_type in [OP.AccessType.RW, OP.AccessType.WRITE]:
                return False

        return True

    def getBaseConfig(self, loop: OP.Loop) -> Dict[str, Any]:
        config = self.target.defaultConfig()

        use_coloring = False

        for arg in loop.args:
            if isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.RW:
                use_coloring = True
                break

        if use_coloring:
            config["atomics"] = False
            config["color2"] = True

        return config

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        kernel_entities = app.findEntities(loop.kernel, program, [])

        assert(len(kernel_entities) == 1)
        kernel_entity = kernel_entities[0]

        dependencies, _ = ftk.extractDependencies([kernel_entity], app, [])

        kernel_entity = copy.deepcopy(kernel_entity)
        dependencies = copy.deepcopy(dependencies)

        for entity in [kernel_entity] + dependencies:
            ftk.fixHydraIO(entity)

        for entity in [kernel_entity] + dependencies:
            ftk.removeExternals(entity)

        ftk.renameConsts(self.lang, [kernel_entity] + dependencies, app, lambda const: f"op2_const_{const}_d")

        def const_rename(const):
            return f"op2_const_{const}_d"

        def match_indirect(arg):
            return isinstance(arg, OP.ArgDat) and arg.map_id is not None

        def match_atomic_inc(arg):
            return arg.access_type == OP.AccessType.INC and config["atomics"]

        def match_gbl(arg):
            return isinstance(arg, OP.ArgGbl)

        ftk.insertAtomicIncs(
            kernel_entity,
            [kernel_entity] + dependencies,
            loop,
            app,
            lambda arg: match_indirect(arg) and match_atomic_inc(arg),
            c_api=True,
        )

        if config["gbl_inc_atomic"]:
            ftk.insertAtomicIncs(
                kernel_entity,
                [kernel_entity] + dependencies,
                loop,
                app,
                lambda arg: match_gbl(arg) and arg.access_type == OP.AccessType.INC,
                c_api=True,
            )

        info = ftk_c.parseInfo([kernel_entity] + dependencies, app, loop, config, const_rename=const_rename)
        setattr(loop, "const_types", info.consts);

        return ftk_c.translate(info)


Scheme.register(FortranCCuda)


class FortranCHip(FortranCCuda):
    target = Target.get("c_hip")

    loop_host_templates = [Path("fortran/c_hip/loop_host.F90.jinja"), Path("fortran/c_hip/loop_host.hip.h.jinja")]
    master_kernel_templates = [Path("fortran/c_hip/master_kernel.F90.jinja"), Path("fortran/c_hip/master_kernel.hip.cpp.jinja")]


Scheme.register(FortranCHip)
