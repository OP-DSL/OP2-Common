import os
from math import ceil

from jinja2 import Environment, FileSystemLoader

import op as OP

# Jinja configuration
env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../resources/templates")),
    lstrip_blocks=True,
    trim_blocks=True,
)


def direct(x, loop: OP.Loop = None) -> bool:
    if isinstance(x, OP.ArgDat) and x.map_id is None:
        return True

    if isinstance(x, OP.Dat) and loop.args[x.arg_id].map_id is None:
        return True

    if isinstance(x, OP.Loop) and len(x.maps) == 0:
        return True

    return False


def indirect(x, loop: OP.Loop = None) -> bool:
    if isinstance(x, OP.ArgDat) and x.map_id is not None:
        return True

    if isinstance(x, OP.Dat) and loop.args[x.arg_id].map_id is not None:
        return True

    if isinstance(x, OP.Loop) and len(x.maps) > 0:
        return True

    return False


env.tests["direct"] = direct
env.tests["indirect"] = indirect

env.tests["soa"] = lambda dat, loop=None: dat.soa

env.tests["opt"] = lambda arg, loop=None: arg.opt

env.tests["dat"] = lambda arg, loop=None: isinstance(arg, OP.ArgDat)
env.tests["gbl"] = lambda arg, loop=None: isinstance(arg, OP.ArgGbl)

env.tests["vec"] = lambda arg, loop=None: isinstance(arg, OP.ArgDat) and arg.map_idx is not None and arg.map_idx < -1

env.tests["read"] = lambda arg, loop=None: arg.access_type == OP.AccessType.READ
env.tests["write"] = lambda arg, loop=None: arg.access_type == OP.AccessType.WRITE
env.tests["read_write"] = lambda arg, loop=None: arg.access_type == OP.AccessType.RW

env.tests["inc"] = lambda arg, loop=None: arg.access_type == OP.AccessType.INC
env.tests["min"] = lambda arg, loop=None: arg.access_type == OP.AccessType.MIN
env.tests["max"] = lambda arg, loop=None: arg.access_type == OP.AccessType.MAX

env.tests["read_or_write"] = lambda arg, loop=None: arg.access_type in [
    OP.AccessType.READ,
    OP.AccessType.WRITE,
    OP.AccessType.RW,
]
env.tests["reduction"] = lambda arg, loop=None: arg.access_type in [
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


def unpack(tup):
    if not isinstance(tup, tuple):
        return tup

    return tup[0]


def test_to_filter(filter_, key=unpack):
    return lambda xs, loop=None: list(filter(lambda x: env.tests[filter_](key(x), loop), xs))  # type: ignore


env.filters["direct"] = test_to_filter("direct")
env.filters["indirect"] = test_to_filter("indirect")

env.filters["soa"] = test_to_filter("soa")

env.filters["opt"] = test_to_filter("opt")

env.filters["dat"] = test_to_filter("dat")
env.filters["gbl"] = test_to_filter("gbl")

env.filters["vec"] = test_to_filter("vec")

env.filters["read"] = test_to_filter("read")
env.filters["write"] = test_to_filter("write")
env.filters["read_write"] = test_to_filter("read_write")

env.filters["inc"] = test_to_filter("inc")
env.filters["min"] = test_to_filter("min")
env.filters["max"] = test_to_filter("max")

env.filters["read_or_write"] = test_to_filter("read_or_write")
env.filters["reduction"] = test_to_filter("reduction")

env.filters["index"] = lambda xs, x: xs.index(x)

env.filters["round_up"] = lambda x, b: b * ceil(x / b)
