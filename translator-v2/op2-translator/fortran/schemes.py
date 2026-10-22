import copy
import traceback
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

import fortran.translator.kernels as ftk
import fortran.translator.kernels_c as ftk_c
import op as OP
from language import Lang
from scheme import Scheme
from store import Application, ParseError, Program
from target import Target
from util import find


def chunkStrategy(strategy: str) -> bool:
    return strategy in ["chunk_atomics", "chunk_colouring"]


# Whether a chunk plan can follow an argument: one map component, and when
# its targets are staged a scalar type the planner knows.
def chunkGroupable(loop: OP.Loop, arg: OP.ArgDat, staged: bool = True) -> bool:
    if arg.map_idx is not None and arg.map_idx < -1:
        return False
    if not staged:
        return True

    typ = loop.dat(arg).typ
    return (isinstance(typ, OP.Float) and typ.size in [32, 64]) or (
        isinstance(typ, OP.Int) and typ.signed and typ.size == 32
    )


def indirectRW(loop: OP.Loop) -> bool:
    return any(isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.RW
               for arg in loop.args)


# A chunked strategy's groups: the arguments its plan follows, and the
# distinct dats among them in first-use order, one group each.
@dataclass(frozen=True)
class ChunkGroups:
    args: List[OP.ArgDat]
    dats: List[int]

    # An argument's group, and its position, which picks its stage word.
    def group(self, arg: OP.ArgDat) -> int:
        return self.dats.index(arg.dat_id)

    def position(self, arg: OP.ArgDat) -> int:
        return self.args.index(arg)

    # Each group's first argument, which names its dat.
    def leaders(self) -> List[OP.ArgDat]:
        return [next(arg for arg in self.args if arg.dat_id == dat) for dat in self.dats]


# A loop's chunked strategies as a scheme plans them: each one's argument
# groups, and the strategies the loop can statically run, in ladder order.
# staged names the chunked strategies whose kernels stage their targets,
# which restricts their arguments to types the planner knows.
@dataclass(frozen=True)
class ChunkStrategies:
    groups: Dict[str, ChunkGroups]
    valid: List[str]

    # chunk_atomics follows the indirect increments; chunk_colouring every
    # indirect argument of a dat the loop updates (INC or RW), reads included.
    @staticmethod
    def of(loop: OP.Loop, staged: FrozenSet[str]) -> "ChunkStrategies":
        indirect = [arg for arg in loop.args if isinstance(arg, OP.ArgDat) and arg.map_id is not None]
        updated = {arg.dat_id for arg in indirect if arg.access_type in [OP.AccessType.INC, OP.AccessType.RW]}

        def grouped(args: List[OP.ArgDat]) -> ChunkGroups:
            return ChunkGroups(args, list(dict.fromkeys(arg.dat_id for arg in args)))

        groups = {
            "chunk_atomics": grouped([arg for arg in indirect if arg.access_type == OP.AccessType.INC]),
            "chunk_colouring": grouped([arg for arg in indirect if arg.dat_id in updated]),
        }

        def groupable(strategy: str) -> bool:
            return all(chunkGroupable(loop, arg, strategy in staged) for arg in groups[strategy].args)

        if len(groups["chunk_colouring"].args) == 0:
            return ChunkStrategies(groups, ["plain"])

        valid = []
        if not any(arg.access_type == OP.AccessType.RW for arg in groups["chunk_colouring"].args):
            if groupable("chunk_atomics"):
                valid.append("chunk_atomics")
            valid.append("atomics")
        if groupable("chunk_colouring"):
            valid.append("chunk_colouring")
        valid.append("colouring")
        return ChunkStrategies(groups, valid)

    # An override's strategies within the valid set, keeping the lowest valid
    # non-chunked strategy as the runtime's floor; without one, every valid
    # strategy.
    def restrict(self, strategies: Optional[List[str]]) -> List[str]:
        if strategies is None:
            return list(self.valid)
        kept = [strategy for strategy in self.valid if strategy in strategies]
        if all(chunkStrategy(strategy) for strategy in kept):
            kept.append(self.valid[-1])
        return kept


# The strategies a loop's kernels are generated for, and their groups.
def setChunkConfig(config: Dict[str, Any], loop: OP.Loop, staged: FrozenSet[str]) -> None:
    chunks = ChunkStrategies.of(loop, staged)
    config["strategies"] = chunks.restrict(config.get("strategies"))
    config["chunk_groups"] = chunks.groups


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

    consts_template = Path("fortran/openmp/consts.F90.jinja")
    loop_host_templates = [Path("fortran/openmp/loop_host.inc.jinja")]
    master_kernel_templates = [Path("fortran/openmp/master_kernel.F90.jinja")]

    # SIMD lanes sharing a read-write target would lose updates.
    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = super().getConfig(loop, config_overrides)
        if indirectRW(loop):
            config["vectorise"] = config["vectorise"] | {"enable": False}
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

        dependencies, _ = ftk.extractDependencies(kernel_entities, app, [])  # TODO: Loop scope

        kernel_entities = copy.deepcopy(kernel_entities)
        dependencies = copy.deepcopy(dependencies)

        ftk.renameConsts(self.lang, kernel_entities + dependencies, app, lambda const: f"op2_const_{const}")

        vectorise = config["vectorise"]
        if not vectorise["enable"] or loop.kernel in vectorise["blacklist"]:
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
            return arg.access_type in [OP.AccessType.MIN, OP.AccessType.MAX] or (not config["gbl_inc_atomic"] and arg.access_type == OP.AccessType.INC)

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

    consts_template = Path("fortran/c_seq/consts.F90.jinja")
    loop_host_templates = [Path("fortran/c_seq/loop_host.F90.jinja"), Path("fortran/c_seq/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("fortran/c_seq/master_kernel.F90.jinja"), Path("fortran/c_seq/master_kernel.cpp.jinja")]

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

        const_rename = None
        if self.lang.user_consts_module is None:
            const_rename = lambda const: f"op2_const_{const}"
            ftk.renameConsts(self.lang, [kernel_entity] + dependencies, app, const_rename)

        info = ftk_c.parseInfo([kernel_entity] + dependencies, app, loop, config, const_rename=const_rename)
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

    staged = frozenset({"chunk_atomics", "chunk_colouring"})

    def getBaseConfig(self, loop: OP.Loop) -> Dict[str, Any]:
        config = self.target.defaultConfig()

        # The C++ schemes share this target and still read these.
        del config["atomics"]
        del config["color2"]
        return config

    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = super().getConfig(loop, config_overrides)
        setChunkConfig(config, loop, self.staged)
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

        def match_gbl(arg):
            return isinstance(arg, OP.ArgGbl)

        # Each strategy's generated code defines op2_inc: atomic or plain.
        ftk.insertAtomicIncs(
            kernel_entity,
            [kernel_entity] + dependencies,
            loop,
            app,
            lambda arg: match_indirect(arg) and arg.access_type == OP.AccessType.INC,
            c_api=True,
            call="op2_inc",
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

    consts_template = Path("fortran/c_hip/consts.F90.jinja")
    loop_host_templates = [Path("fortran/c_hip/loop_host.F90.jinja"), Path("fortran/c_hip/loop_host.hip.h.jinja")]
    master_kernel_templates = [Path("fortran/c_hip/master_kernel.F90.jinja"), Path("fortran/c_hip/master_kernel.hip.cpp.jinja")]


Scheme.register(FortranCHip)


class FortranCOpenMP(Scheme):
    lang = Lang.get("F90")
    target = Target.get("c_openmp")

    fallback = Scheme.get((Lang.get("F90"), Target.get("seq")))

    consts_template = Path("fortran/c_openmp/consts.F90.jinja")
    loop_host_templates = [Path("fortran/c_openmp/loop_host.F90.jinja"), Path("fortran/c_openmp/loop_host.hpp.jinja")]
    master_kernel_templates = [Path("fortran/c_openmp/master_kernel.F90.jinja"), Path("fortran/c_openmp/master_kernel.cpp.jinja")]

    def canGenLoopHost(self, loop: OP.Loop) -> bool:
        for arg in loop.args:
            if isinstance(arg, OP.ArgGbl) and arg.access_type in [OP.AccessType.RW, OP.AccessType.WRITE]:
                return False

        return True

    # chunk_colouring runs the plain body over its chunks.
    staged = frozenset({"chunk_atomics"})

    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = super().getConfig(loop, config_overrides)
        setChunkConfig(config, loop, self.staged)
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

        const_rename = None
        if self.lang.user_consts_module is None:
            const_rename = lambda const: f"op2_const_{const}"
            ftk.renameConsts(self.lang, [kernel_entity] + dependencies, app, const_rename)

        # Each strategy's generated code defines op2_inc: atomic or plain.
        ftk.insertAtomicIncs(
            kernel_entity,
            [kernel_entity] + dependencies,
            loop,
            app,
            lambda arg: isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.INC,
            c_api=True,
            call="op2_inc",
        )

        info = ftk_c.parseInfo([kernel_entity] + dependencies, app, loop, config, const_rename=const_rename)
        return ftk_c.translate(info)


Scheme.register(FortranCOpenMP)
