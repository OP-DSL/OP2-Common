
# Standard library imports
from typing import Tuple, Dict, List
from pathlib import Path
import json
import os

# Third party imports
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template

# Local application imports
import op as OP


# Jinja configuration
env = Environment(
  loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), '../resources/templates')),
  lstrip_blocks=True,
  trim_blocks=True,
)

env.tests['r_or_w_acc'] = lambda arg: arg.acc in (OP.READ, OP.WRITE)
env.tests['rw_acc'] = lambda arg: arg.acc == OP.RW
env.tests['inc_acc'] = lambda arg: arg.acc == OP.INC
env.tests['without_dim'] = lambda arg: not isinstance(arg.dim, int) 
env.tests['global'] = lambda arg: arg.global_
env.tests['direct'] = lambda arg: arg.direct
env.tests['indirect'] = lambda arg: arg.indirect

