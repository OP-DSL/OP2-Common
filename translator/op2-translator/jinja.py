import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

import op as OP
from scheme import LoopHost

# Jinja configuration
env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../resources/templates")),
    lstrip_blocks=True,
    trim_blocks=True,
)


def direct(x: Union[OP.Arg, LoopHost]) -> bool:
    if isinstance(x, OP.ArgDat) and x.map_ptr is None:
        return True

    if isinstance(x, LoopHost) and len(x.maps) == 0:
        return True

    return False


def indirect(x: Union[OP.Arg, LoopHost]) -> bool:
    if isinstance(x, OP.ArgDat) and x.map_ptr is not None:
        return True

    if isinstance(x, LoopHost) and len(x.maps) > 0:
        return True

    return False


env.tests["direct"] = direct
env.tests["indirect"] = indirect

env.tests["soa"] = lambda dat: dat.soa

env.tests["opt"] = lambda arg: arg.opt

env.tests["dat"] = lambda arg: isinstance(arg, OP.ArgDat)
env.tests["gbl"] = lambda arg: isinstance(arg, OP.ArgGbl)

env.tests["vec"] = lambda arg: isinstance(arg, OP.ArgDat) and arg.map_idx is not None and arg.map_idx < -1

env.tests["read"] = lambda arg: arg.access_type == OP.AccessType.READ
env.tests["write"] = lambda arg: arg.access_type == OP.AccessType.WRITE
env.tests["read_write"] = lambda arg: arg.access_type == OP.AccessType.RW

env.tests["inc"] = lambda arg: arg.access_type == OP.AccessType.INC
env.tests["min"] = lambda arg: arg.access_type == OP.AccessType.MIN
env.tests["max"] = lambda arg: arg.access_type == OP.AccessType.MAX

env.tests["read_or_write"] = lambda arg: arg.access_type in [OP.AccessType.READ, OP.AccessType.WRITE, OP.AccessType.RW]
env.tests["reduction"] = lambda arg: arg.access_type in [OP.AccessType.INC, OP.AccessType.MIN, OP.AccessType.MAX]


def read_in(dat: OP.Dat, loop_host: LoopHost) -> bool:
    for arg, idx in loop_host.args:
        if not isinstance(arg, OP.ArgDat):
            continue

        if arg.dat_ptr == dat.ptr and arg.access_type != OP.AccessType.READ:
            return False

    return True


env.tests["read_in"] = read_in


def unpack_arg(arg):
    if isinstance(arg, OP.Arg):
        return arg

    return arg[0]


def unpack_dat(dat):
    if isinstance(dat, OP.Dat):
        return dat

    return dat[0]


def test_to_filter(filter_, key=unpack_arg):
    return lambda xs: list(filter(lambda x: env.tests[filter_](key(x)), xs))  # type: ignore


env.filters["direct"] = test_to_filter("direct")
env.filters["indirect"] = test_to_filter("indirect")

env.filters["soa"] = test_to_filter("soa", unpack_dat)

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
