# The execution strategies a loop's generated code can run, and the argument
# groups of the chunked ones, shared by every language's schemes.
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional

import op as OP


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
