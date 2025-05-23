import os
from math import ceil
from typing import Optional

from jinja2 import Environment, FileSystemLoader, pass_context

import op as OP
from fortran.translator.kernels_c import FArray, FCharacter

# Jinja configuration
env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../resources/templates")),
    lstrip_blocks=True,
    trim_blocks=True,
)

env.tests["farray"] = lambda type_: isinstance(type_, FArray)
env.tests["fcharacter"] = lambda type_: isinstance(type_, FCharacter) or (isinstance(type_, FArray) and isinstance(type_.inner, FCharacter))

def direct(x, loop: Optional[OP.Loop] = None) -> bool:
    if isinstance(x, (OP.ArgDat, OP.ArgIdx)) and x.map_id is None:
        return True

    if isinstance(x, OP.Dat):
        assert loop is not None

        arg = loop.args[x.arg_id]
        assert isinstance(arg, OP.ArgDat)

        return arg.map_id is None

    if isinstance(x, OP.Loop) and len(x.maps) == 0:
        return True

    return False


def indirect(x, loop: Optional[OP.Loop] = None) -> bool:
    if isinstance(x, (OP.ArgDat, OP.ArgIdx)) and x.map_id is not None:
        return True

    if isinstance(x, OP.Dat):
        assert loop is not None

        arg = loop.args[x.arg_id]
        assert isinstance(arg, OP.ArgDat)

        return arg.map_id is not None

    if isinstance(x, OP.Loop) and len(x.maps) != 0:
        return True

    return False


env.tests["direct"] = direct
env.tests["indirect"] = indirect


def soa(x, loop: Optional[OP.Loop] = None) -> bool:
    if isinstance(x, OP.ArgDat):
        assert loop is not None

    if isinstance(x, OP.ArgDat) and loop is not None:
        return loop.dats[x.dat_id].soa

    if isinstance(x, OP.Dat):
        return x.soa

    return False


env.tests["soa"] = soa


def scalar(x, loop: Optional[OP.Loop] = None) -> bool:
    if isinstance(x, OP.ArgDat):
        assert loop is not None

    if isinstance(x, OP.ArgDat) and loop is not None:
        return loop.dats[x.dat_id].dim == 1

    if isinstance(x, OP.Dat):
        return x.dim == 1

    if isinstance(x, (OP.ArgGbl, OP.ArgInfo)):
        return x.dim == 1

    assert False


env.tests["scalar"] = scalar
env.tests["multidim"] = lambda arg, loop=None: not scalar(arg, loop)


env.tests["opt"] = lambda arg, loop=None: hasattr(arg, "opt") and arg.opt

env.tests["dat"] = lambda arg, loop=None: isinstance(arg, OP.ArgDat)
env.tests["gbl"] = lambda arg, loop=None: isinstance(arg, OP.ArgGbl)
env.tests["idx"] = lambda arg, loop=None: isinstance(arg, OP.ArgIdx)
env.tests["info"] = lambda arg, loop=None: isinstance(arg, OP.ArgInfo)

env.tests["vec"] = lambda arg, loop=None: isinstance(arg, OP.ArgDat) and arg.map_idx is not None and arg.map_idx < -1
env.tests["runtime_map_idx"] = (
    lambda arg, loop=None: isinstance(arg, OP.ArgDat) and arg.map_id is not None and arg.map_idx is None
)

env.tests["read"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == OP.AccessType.READ
env.tests["write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == OP.AccessType.WRITE
env.tests["read_write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == OP.AccessType.RW

env.tests["inc"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == OP.AccessType.INC
env.tests["min"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == OP.AccessType.MIN
env.tests["max"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == OP.AccessType.MAX

env.tests["work"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type == OP.AccessType.WORK

env.tests["read_or_write"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type in [
    OP.AccessType.READ,
    OP.AccessType.WRITE,
    OP.AccessType.RW,
]

env.tests["min_or_max"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type in [
    OP.AccessType.MIN,
    OP.AccessType.MAX,
]

env.tests["reduction"] = lambda arg, loop=None: hasattr(arg, "access_type") and arg.access_type in [
    OP.AccessType.INC,
    OP.AccessType.MIN,
    OP.AccessType.MAX,
]


def read_in(dat: OP.Dat, loop: OP.Loop) -> bool:
    for arg in loop.args:
        if not isinstance(arg, OP.ArgDat):
            continue

        if arg.dat_id == dat.id and arg.access_type != OP.AccessType.READ:
            return False

    return True


env.tests["read_in"] = read_in
env.tests["instance"] = lambda x, c: isinstance(x, c)


@pass_context
def select2_filter(context, xs, *tests):
    ys = []
    for x in xs:
        for test in tests:
            if context.environment.call_test(test, x):
                ys.append(x)
                break

    return ys

env.filters["select2"] = select2_filter


def unpack(tup):
    if not isinstance(tup, tuple):
        return tup

    return tup[0]


def test_to_filter(filter_, key=unpack):
    return lambda xs, loop=None: list(filter(lambda x: env.tests[filter_](key(x), loop), xs))  # type: ignore


env.filters["direct"] = test_to_filter("direct")
env.filters["indirect"] = test_to_filter("indirect")

env.filters["soa"] = test_to_filter("soa")

env.filters["scalar"] = test_to_filter("scalar")
env.filters["multidim"] = test_to_filter("multidim")

env.filters["opt"] = test_to_filter("opt")

env.filters["dat"] = test_to_filter("dat")
env.filters["gbl"] = test_to_filter("gbl")
env.filters["idx"] = test_to_filter("idx")
env.filters["info"] = test_to_filter("info")

env.filters["vec"] = test_to_filter("vec")
env.filters["runtime_map_idx"] = test_to_filter("runtime_map_idx")

env.filters["read"] = test_to_filter("read")
env.filters["write"] = test_to_filter("write")
env.filters["read_write"] = test_to_filter("read_write")

env.filters["inc"] = test_to_filter("inc")
env.filters["min"] = test_to_filter("min")
env.filters["max"] = test_to_filter("max")

env.filters["work"] = test_to_filter("work")
env.filters["read_or_write"] = test_to_filter("read_or_write")
env.filters["min_or_max"] = test_to_filter("min_or_max")

env.filters["reduction"] = test_to_filter("reduction")

env.filters["index"] = lambda xs, x: xs.index(x)

env.filters["round_up"] = lambda x, b: b * ceil(x / b)
