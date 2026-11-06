# What a loop's arguments allow of the ways its generated code can run,
# shared by every language's schemes.
import op as OP


def indirectRW(loop: OP.Loop) -> bool:
    return any(isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.access_type == OP.AccessType.RW
               for arg in loop.args)
