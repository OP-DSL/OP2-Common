
# Standard library imports
from typing import Optional, List, Set
from pathlib import Path
import re
import os

# Third party imports
from clang.cindex import Index, Config, TranslationUnit, Cursor, CursorKind

# Local application imports
from store import ParseError, Program, Kernel, Location
from util import enumRegex, safeFind
import op as OP


macro_instances = {} # TODO: Cleanup


def parseKernel(self, path: Path, name: str) -> Kernel:
  # Invoke Clang parser on kernel source
  translation_unit = Index.create().parse(path)

  # Collect root-level nodes
  nodes = translation_unit.cursor.get_children()

  # Search for kernel function
  node = safeFind(nodes, lambda n: n.kind == CursorKind.FUNCTION_DECL and n.spelling == name)
  if not node:
    raise ParseError(f'failed to locate kernel function {name}', parseLocation(node))

  # Collect parameter types
  params = []
  for n in node.get_children():
    if n.kind == CursorKind.PARM_DECL:
      type = n.type.get_pointee() or n.type
      type = re.sub(r'\s*const\s*', '', type.spelling)
      param = (n.spelling, type)
      params.append(param)

  return Kernel(name, path, translation_unit.cursor, params)


def parseProgram(self, path: Path, include_dirs: Set[Path]) -> Program:
  # Locate OP2 install
  op2_install = os.getenv('OP2_INSTALL_PATH')
  if not op2_install:
    exit('Fatal: OP2_INSTALL_PATH not set')

  # Add OP2 includes
  op2_include = Path(op2_install).joinpath('c/include')
  include_dirs.add(op2_include)

  # Form Clang args
  args = [ f'-I{dir}' for dir in include_dirs ]

  # Invoke Clang parser on the program source
  translation_unit = Index.create().parse(
    path,
    args=args,
    options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
  )

  # Throw the parse error first parse error caught in the diagnostics
  error = next(iter(translation_unit.diagnostics), None)
  if error:
    raise ParseError(error.spelling, parseLocation(error))

  # Initialise search stack
  program = Program(path)
  stack = []

  for child in translation_unit.cursor.get_children():
    # Ignore macro definitions and cursors outside of the program file
    if child.kind != CursorKind.MACRO_DEFINITION and child.location.file.name == translation_unit.spelling:
      # Collect the locations and identifiers of macro instances
      if child.kind == CursorKind.MACRO_INSTANTIATION:
        macro_instances[(child.location.line, child.location.column)] = child.displayname
      # Populate with the top-level cursors from the target file
      else:
        stack.append(child)

  ptr: str

  # DFS
  while stack:
    # Manage the stack
    node = stack.pop()
    stack.extend(node.get_children())

    # TODO: This is a bit of a hack. Cleanup
    if node.kind == CursorKind.VAR_DECL:
      if safeFind(node.get_children(), lambda n: n.kind == CursorKind.CALL_EXPR):
        ptr = node.spelling

    # Focus on function calls
    if node.kind == CursorKind.CALL_EXPR:
      name = node.spelling
      args = list(node.get_children())[1:]
      loc = parseLocation(node)

      if name == 'op_init':
        program.recordInit(loc)

      elif name == 'op_decl_set':
        program.sets.append(parseSet(args, ptr, loc))

      elif name == 'op_decl_map':
        program.maps.append(parseMap(args, ptr, loc))

      elif name == 'op_decl_dat':
        program.datas.append(parseDat(args, ptr, loc))

      elif name == 'op_decl_const':
        program.consts.append(parseConst(args, loc))

      elif name == 'op_par_loop':
        program.loops.append(parseLoop(args, loc))

      elif name == 'op_exit':
        program.recordExit()

  return program


def parseSet(nodes: List[Cursor], ptr: str, loc: Location) -> OP.Set:
  if len(nodes) != 2:
    raise ParseError('incorrect number of nodes passed to op_decl_set', loc)

  _     = parseIdentifier(nodes[0])
  debug = parseStringLit(nodes[1])

  return OP.Set(ptr)


def parseMap(nodes: List[Cursor], ptr: str, loc: Location) -> OP.Map:
  if len(nodes) != 5:
    raise ParseError('incorrect number of args passed to op_decl_map', loc)

  from_set = parseIdentifier(nodes[0])
  to_set   = parseIdentifier(nodes[1])
  dim      = parseIntLit(nodes[2], signed=False)
  _        = parseIdentifier(nodes[3])
  debug     = parseStringLit(nodes[4])

  return OP.Map(from_set, to_set, dim, ptr, loc)


def parseDat(nodes: List[Cursor], ptr: str, loc: Location) -> OP.Data:
  if len(nodes) != 5:
    raise ParseError('incorrect number of args passed to op_decl_dat', loc)

  set_  = parseIdentifier(nodes[0])
  dim   = parseIntLit(nodes[1], signed=False)
  typ   = parseStringLit(nodes[2])
  _     = parseIdentifier(nodes[3])
  debug = parseStringLit(nodes[4])

  return OP.Data(set_, dim, typ, ptr, loc)


def parseConst(nodes: List[Cursor], loc: Location) -> OP.Const:
  if len(nodes) != 3:
    raise ParseError('incorrect number of args passed to op_decl_const', loc)

  # TODO dim may not be literal
  dim  = parseIntLit(nodes[0], signed=False)
  typ  = parseStringLit(nodes[1])
  ptr = parseIdentifier(nodes[2])
  debug = ptr

  return OP.Const(ptr, dim, typ, debug, loc)


def parseLoop(nodes: List[Cursor], loc: Location) -> OP.Loop:
  if len(nodes) < 3:
    raise ParseError('incorrect number of args passed to op_par_loop')

  # Parse loop kernel and set
  kernel = parseIdentifier(nodes[0])
  _  = parseStringLit(nodes[1])
  set_ = parseIdentifier(nodes[2])

  loop_args = []

  # Parse loop args
  for node in nodes[3:]:
    node = descend(descend(node))
    name = node.spelling
    arg_loc = parseLocation(node)
    args = list(node.get_children())[1:]

    if name == 'op_arg_dat':
      loop_args.append(parseArgDat(args, arg_loc))

    elif name == 'op_opt_arg_dat':
      loop_args.append(parseOptArgDat(args, arg_loc))

    elif name == 'op_arg_gbl':
      loop_args.append(parseArgGbl(args, arg_loc))

    elif name == 'op_opt_arg_gbl':
      loop_args.append(parseOptArgGbl(args, arg_loc))

    else:
      raise ParseError(f'invalid loop argument {name}', parseLocation(node))

  return OP.Loop(kernel, set_, loc, loop_args)


def parseArgDat(nodes: List[Cursor], loc: Location) -> OP.Arg:
  if len(nodes) != 6:
    raise ParseError('incorrect number of args passed to op_arg_dat', loc)

  access_regex = enumRegex(OP.DAT_ACCESS_TYPES)

  var  = parseIdentifier(nodes[0])
  idx  = parseIntLit(nodes[1], signed=True)
  map_ = parseIdentifier(nodes[2]) or OP.ID
  dim  = parseIntLit(nodes[3], signed=False)
  typ  = parseStringLit(nodes[4])
  acc  = macro_instances[(nodes[5].location.line, nodes[5].location.column)] # TODO: Cleanup

  return OP.Arg(var, dim, typ, acc, loc, map_, idx)


def parseOptArgDat(nodes: List[Cursor], loc: Location) -> OP.Arg:
  if len(nodes) != 7:
    ParseError('incorrect number of args passed to op_opt_arg_dat', loc)

  # Parse opt argument
  opt = parseIdentifier(nodes[0])

  # Parse standard argDat arguments
  dat = parseArgDat(nodes[1:], loc)

  # Return augmented dat
  dat.opt = opt
  return dat


def parseArgGbl(nodes: List[Cursor], loc: Location) -> OP.Arg:
  if len(nodes) != 4:
    raise ParseError('incorrect number of args passed to op_arg_gbl', loc)

  access_regex = enumRegex(OP.GBL_ACCESS_TYPES)

  var = parseIdentifier(nodes[0])
  dim = parseIntLit(nodes[1], signed=False)
  typ = parseStringLit(nodes[2])
  acc = macro_instances[(nodes[3].location.line, nodes[3].location.column)] # TODO: Cleanup

  return OP.Arg(var, dim, typ, acc, loc)


def parseOptArgGbl(nodes: List[Cursor], loc: Location) -> OP.Arg:
  if len(nodes) != 5:
    raise ParseError('incorrect number of args passed to op_opt_arg_gbl', loc)

  # Parse opt argument
  opt = parseIdentifier(nodes[0])

  # Parse standard argGbl arguments
  dat = parseArgGbl(nodes[1:], loc)

  # Return augmented dat
  dat.opt = opt
  return dat


def parseIdentifier(node: Cursor, regex: str = None) -> str:
  # TODO: Check this
  while node.kind == CursorKind.CSTYLE_CAST_EXPR:
    node = list(node.get_children())[1]

  # Descend to child node
  if node.kind == CursorKind.UNEXPOSED_EXPR:
    node = descend(node)

  # Descend to child node
  if node.kind == CursorKind.UNARY_OPERATOR and next(node.get_tokens()).spelling in ('&', '*'):
    node = descend(node)

  # Check for null
  if node.kind == CursorKind.GNU_NULL_EXPR:
    return ''

  # Validate the node
  if node.kind != CursorKind.DECL_REF_EXPR:
    raise ParseError('expected identifier')

  value = node.spelling

  # Apply conditional regex constraint
  if regex and not re.match(regex, value):
    raise ParseError(f'expected identifier matching {regex}')

  return value


def parseIntLit(node: Cursor, signed: bool = True) -> int:
  # Assume the literal is not negated
  negation = False

  # Check if the node is wrapped in a valid unary negation
  if signed and node.kind == CursorKind.UNARY_OPERATOR and next(node.get_tokens()).spelling == '-':
    negation = True
    node = descend(node)

  # Validate the node
  if node.kind != CursorKind.INTEGER_LITERAL:
    if not signed:
      raise ParseError('expected unsigned integer literal')
    else:
      raise ParseError('expected integer literal')

  # Extract the value
  value = int(next(node.get_tokens()).spelling)

  return -value if negation else value


def parseStringLit(node: Cursor, regex: str = None) -> str:
  # Validate the node
  if node.kind != CursorKind.UNEXPOSED_EXPR:
    raise ParseError('expected string literal')

  # Descend to child node
  node = descend(node)

  # Validate the node
  if node.kind != CursorKind.STRING_LITERAL:
    raise ParseError('expected string literal')

  # Extract value from string delimeters
  value = node.spelling[1:-1]

  # Apply conditional regex constraint
  if regex and not re.match(regex, value):
    raise ParseError(f'expected string literal matching {regex}')

  return value


def parseLocation(node: Cursor) -> Location:
  return Location(
    node.location.file.name,
    node.location.line,
    node.location.column
  )


def descend(node: Cursor) -> Optional[Cursor]:
  return next(node.get_children(), None)
