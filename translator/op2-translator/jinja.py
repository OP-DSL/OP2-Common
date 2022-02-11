import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape

import op as OP

# Jinja configuration
env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "../resources/templates")),
    lstrip_blocks=True,
    trim_blocks=True,
)

env.tests["direct"] = lambda lh: len(lh.maps) == 0
env.tests["indirect"] = lambda lh: len(lh.maps) > 0

env.tests["dat"] = lambda arg: isinstance(arg, OP.ArgDat)
env.tests["gbl"] = lambda arg: isinstance(arg, OP.ArgGbl)

env.tests["vec"] = lambda arg: isinstance(arg, OP.ArgDat) and arg.map_idx is not None and arg.map_idx < -1

env.tests["reduction"] = lambda access_type: access_type in [OP.AccessType.INC, OP.AccessType.MIN, OP.AccessType.MAX]
