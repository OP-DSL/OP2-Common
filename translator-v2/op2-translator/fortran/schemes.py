import copy
import traceback
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import fortran.flang_kernels as fk
import fortran.flang_kernels_c as fk_c
import fortran.flang_writer as fwriter
import fortran.translator.kernels as ftk
import fortran.translator.kernels_c as ftk_c
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, Function, ParseError, Program
from target import Target
from util import find


# schemes that have already warned the user that they
# fall back to fparser2 under --parser flang
_FLANG_FALLBACK_WARNED: set = set()


def _all_entities_have_flang_source(entities: Iterable) -> bool:
    """
    Return True if every entity is a Function and carries a flang_source.
    """
    saw_any = False
    for e in entities:
        if not isinstance(e, Function):
            return False
        if not getattr(e, "flang_source", None):
            return False
        saw_any = True
    return saw_any


def _use_flang_kernels_c(lang: Lang, entities: Iterable) -> bool:
    """
    Return True if --parser flang is selected and every entity carries flang_body.
    """
    if getattr(lang, "requested_parser", "fparser2") != "flang":
        return False

    return fk_c.canTranslateWithFlang(list(entities))


def _warn_flang_fallback_once(lang: Lang, scheme_name: str) -> None:
    if getattr(lang, "requested_parser", "fparser2") != "flang":
        return
    if scheme_name in _FLANG_FALLBACK_WARNED:
        return
    print(
        f"Warning: --parser flang does not yet drive kernel translation for "
        f"{scheme_name}; falling back to the fparser2 path for this scheme.",
        file=sys.stderr,
    )
    _FLANG_FALLBACK_WARNED.add(scheme_name)


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

        all_entities = kernel_entities + dependencies

        use_flang = (
            getattr(self.lang, "requested_parser", "fparser2") == "flang"
            and _all_entities_have_flang_source(all_entities)
        )

        if use_flang:
            if self.lang.user_consts_module is None:
                fwriter.rename_consts(
                    self.lang, all_entities, app, lambda const: f"op2_const_{const}"
                )
            return fwriter.write_source(all_entities)

        if self.lang.user_consts_module is None:
            ftk.renameConsts(
                self.lang, all_entities, app, lambda const: f"op2_const_{const}"
            )
        return ftk.writeSource(all_entities)


Scheme.register(FortranSeq)


class FortranOpenMP(Scheme):
    lang = Lang.get("F90")
    target = Target.get("openmp")

    fallback = Scheme.get((Lang.get("F90"), Target.get("seq")))

    consts_template = Path("fortran/openmp/consts.F90.jinja")
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

        all_entities = kernel_entities + dependencies

        def match_indirect(arg):
            return isinstance(arg, OP.ArgDat) and arg.map_id is not None

        def match_gbl_reduction(arg):
            return isinstance(arg, OP.ArgGbl) and arg.access_type in [
                OP.AccessType.INC,
                OP.AccessType.MIN,
                OP.AccessType.MAX,
            ]

        use_flang = (
            getattr(self.lang, "requested_parser", "fparser2") == "flang"
            and _all_entities_have_flang_source(all_entities)
        )

        if use_flang:
            fwriter.rename_consts(
                self.lang, all_entities, app, lambda const: f"op2_const_{const}"
            )

            if not config["vectorise"]:
                return fwriter.write_source(all_entities)

            simd_kernel_entities = copy.deepcopy(kernel_entities)
            fwriter.rename_entities(simd_kernel_entities, lambda name: f"{name}_simd")

            for simd_kernel_entity in simd_kernel_entities:
                fwriter.insert_strides(
                    [simd_kernel_entity] + dependencies,
                    loop,
                    lambda arg: "SIMD_LEN",
                    match=lambda arg: match_indirect(arg) or match_gbl_reduction(arg),
                )

            return fwriter.write_source(
                kernel_entities + simd_kernel_entities + dependencies
            )

        _warn_flang_fallback_once(self.lang, f"Fortran/{self.target.name}")

        ftk.renameConsts(self.lang, all_entities, app, lambda const: f"op2_const_{const}")

        if not config["vectorise"]:
            return ftk.writeSource(all_entities)

        simd_kernel_entities = copy.deepcopy(kernel_entities)
        ftk.renameEntities(simd_kernel_entities, lambda name: f"{name}_simd")

        for simd_kernel_entity in simd_kernel_entities:
            ftk.insertStrides(
                simd_kernel_entity,
                [simd_kernel_entity] + dependencies,
                loop,
                app,
                lambda arg: "SIMD_LEN",
                match=lambda arg: match_indirect(arg) or match_gbl_reduction(arg),
            )

        return ftk.writeSource(kernel_entities + simd_kernel_entities + dependencies)


Scheme.register(FortranOpenMP)


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

        all_entities = kernel_entities + dependencies

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
            return arg.access_type in [OP.AccessType.MIN, OP.AccessType.MAX] or (not config["gbl_inc_atomic"] and arg.access_type == OP.AccessType.INC)

        def match_work(arg):
            return arg.access_type == OP.AccessType.WORK

        use_flang = (
            getattr(self.lang, "requested_parser", "fparser2") == "flang"
            and _all_entities_have_flang_source(all_entities)
        )

        if use_flang:
            fwriter.rename_consts(
                self.lang, all_entities, app, lambda const: f"op2_const_{const}_d"
            )

            for entity in all_entities:
                fwriter.fix_hydra_io(entity)
                fwriter.remove_externals(entity)

            modified = fwriter.insert_strides(
                all_entities,
                loop,
                lambda arg: "direct",
                lambda arg: match_soa(arg) and not match_indirect(arg),
            )

            modified = fwriter.insert_strides(
                all_entities,
                loop,
                lambda arg: f"dat{arg.dat_id}",
                lambda arg: match_soa(arg) and match_indirect(arg),
                modified,
            )

            modified = fwriter.insert_strides(
                all_entities,
                loop,
                lambda arg: "gbl",
                lambda arg: (match_gbl(arg) and (match_reduction(arg) or match_work(arg))) or match_info(arg),
                modified,
            )

            fwriter.insert_atomic_incs(
                all_entities,
                loop,
                lambda arg: match_indirect(arg) and match_atomic_inc(arg),
            )

            if config["gbl_inc_atomic"]:
                fwriter.insert_atomic_incs(
                    all_entities,
                    loop,
                    lambda arg: match_gbl(arg) and arg.access_type == OP.AccessType.INC,
                )

            return fwriter.write_source(all_entities, "attributes(device) &\n")

        _warn_flang_fallback_once(self.lang, f"Fortran/{self.target.name}")

        ftk.renameConsts(self.lang, all_entities, app, lambda const: f"op2_const_{const}_d")

        for entity in all_entities:
            ftk.fixHydraIO(entity)

        for entity in all_entities:
            ftk.removeExternals(entity)

        modified = ftk.insertStrides(
            kernel_entities[0],
            all_entities,
            loop,
            app,
            lambda arg: f"direct",
            lambda arg: match_soa(arg) and not match_indirect(arg),
        )

        modified = ftk.insertStrides(
            kernel_entities[0],
            all_entities,
            loop,
            app,
            lambda arg: f"dat{arg.dat_id}",
            lambda arg: match_soa(arg) and match_indirect(arg),
            modified,
        )

        modified = ftk.insertStrides(
            kernel_entities[0],
            all_entities,
            loop,
            app,
            lambda arg: f"gbl",
            lambda arg: (match_gbl(arg) and (match_reduction(arg) or match_work(arg))) or match_info(arg),
            modified,
        )

        ftk.insertAtomicIncs(
            kernel_entities[0],
            all_entities,
            loop,
            app,
            lambda arg: match_indirect(arg) and match_atomic_inc(arg),
        )

        if config["gbl_inc_atomic"]:
            ftk.insertAtomicIncs(
                kernel_entities[0],
                all_entities,
                loop,
                app,
                lambda arg: match_gbl(arg) and arg.access_type == OP.AccessType.INC,
            )

        return ftk.writeSource(all_entities, "attributes(device) &\n")


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

        all_entities = [kernel_entity] + dependencies

        if _use_flang_kernels_c(self.lang, all_entities):
            fk.fix_hydra_io(all_entities)
            info = fk_c.parseInfo(all_entities, app, loop, config)
            return fk_c.translate(info)

        _warn_flang_fallback_once(self.lang, f"Fortran/{self.target.name}")

        for entity in all_entities:
            ftk.fixHydraIO(entity)

        for entity in all_entities:
            ftk.removeExternals(entity)

        info = ftk_c.parseInfo(all_entities, app, loop, config)
        return ftk_c.translate(info)


Scheme.register(FortranCSeq)


class FortranCCuda(Scheme):
    lang = Lang.get("F90")
    target = Target.get("c_cuda")

    fallback = Scheme.get((Lang.get("F90"), Target.get("seq")))

    consts_template = Path("fortran/c_cuda/consts.F90.jinja")
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

        all_entities = [kernel_entity] + dependencies

        def const_rename(const):
            return f"op2_const_{const}_d"

        def match_indirect(arg):
            return isinstance(arg, OP.ArgDat) and arg.map_id is not None

        def match_atomic_inc(arg):
            return arg.access_type == OP.AccessType.INC and config["atomics"]

        def match_gbl(arg):
            return isinstance(arg, OP.ArgGbl)

        if _use_flang_kernels_c(self.lang, all_entities):
            fk.fix_hydra_io(all_entities)
            fk.rename_consts(all_entities, app.constPtrs(), const_rename)

            fk.insert_atomic_incs(
                all_entities,
                loop,
                lambda arg: match_indirect(arg) and match_atomic_inc(arg),
            )

            if config["gbl_inc_atomic"]:
                fk.insert_atomic_incs(
                    all_entities,
                    loop,
                    lambda arg: match_gbl(arg) and arg.access_type == OP.AccessType.INC,
                )

            info = fk_c.parseInfo(all_entities, app, loop, config, const_rename=const_rename)
            setattr(loop, "const_types", info.consts)

            return fk_c.translate(info)

        _warn_flang_fallback_once(self.lang, f"Fortran/{self.target.name}")

        for entity in all_entities:
            ftk.fixHydraIO(entity)

        for entity in all_entities:
            ftk.removeExternals(entity)

        ftk.renameConsts(self.lang, all_entities, app, const_rename)

        ftk.insertAtomicIncs(
            kernel_entity,
            all_entities,
            loop,
            app,
            lambda arg: match_indirect(arg) and match_atomic_inc(arg),
            c_api=True,
        )

        if config["gbl_inc_atomic"]:
            ftk.insertAtomicIncs(
                kernel_entity,
                all_entities,
                loop,
                app,
                lambda arg: match_gbl(arg) and arg.access_type == OP.AccessType.INC,
                c_api=True,
            )

        info = ftk_c.parseInfo(all_entities, app, loop, config, const_rename=const_rename)
        setattr(loop, "const_types", info.consts);

        return ftk_c.translate(info)


Scheme.register(FortranCCuda)


class FortranCHip(FortranCCuda):
    target = Target.get("c_hip")

    consts_template = Path("fortran/c_hip/consts.F90.jinja")
    loop_host_templates = [Path("fortran/c_hip/loop_host.F90.jinja"), Path("fortran/c_hip/loop_host.hip.h.jinja")]
    master_kernel_templates = [Path("fortran/c_hip/master_kernel.F90.jinja"), Path("fortran/c_hip/master_kernel.hip.cpp.jinja")]


Scheme.register(FortranCHip)
