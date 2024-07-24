from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import fparser.two.Fortran2003 as f2003
import fparser.two.Fortran2008 as f2008
import fparser.two.utils as fpu

import fortran.util as fu
import op as OP
from language import Lang
from op import OpError
from store import Application, Entity, Function, Program
from util import find, safeFind


def indent(src: str, width: int = 4) -> str:
    indent = " " * width

    lines = src.splitlines()
    for i in range(len(lines)):
        if len(lines[i]) == 0 or lines[i].isspace():
            continue

        lines[i] = indent + lines[i]

    return "\n".join(lines)


class FType(ABC):
    @abstractmethod
    def asLocal(self, var: str) -> str:
        pass


class FPrimitive(FType, ABC):
    def asLocal(self, var: str) -> str:
        return f"{self.asC()} {var}"

    @abstractmethod
    def asC(self) -> str:
        pass


@dataclass(frozen=True)
class FArray(FType):
    shape: List[Tuple[str, str]]
    inner: FPrimitive

    def asLocal(self, var: str) -> str:
        sizes = []
        for lb, ub in self.shape:
            if lb == "1":
                sizes.append(f"({ub})")
            else:
                sizes.append(f"(1 + {ub} - ({lb}))")

        return f"{self.inner.asC()} {var}[{'*'.join(sizes)}]"

    def spanBounds(self) -> str:
        bounds = []
        for lb, ub in self.shape:
            bounds.append(f"f2c::Extent{{{lb}, {ub}}}")

        return ", ".join(bounds)

    def asSpan(self, ptr: str, var: str, ctx: 'Context') -> str:
        return f"f2c::Span {var}{{{ptr}, {self.spanBounds()}}}"


@dataclass(frozen=True)
class FInteger(FPrimitive):
    kind: int

    def asC(self) -> str:
        if self.kind == 4:
            return "int"
        elif self.kind == 8:
            return "int64_t"
        else:
            raise OpError(f"FInteger unknown kind: {self.kind}")


@dataclass(frozen=True)
class FReal(FPrimitive):
    kind: int

    def asC(self) -> str:
        if self.kind == 4:
            return "float"
        elif self.kind == 8:
            return "double"
        else:
            raise OpError(f"FReal unknown kind: {self.kind}")


@dataclass(frozen=True)
class FLogical(FPrimitive):
    def asC(self) -> str:
        return "bool"


@dataclass(frozen=True)
class FCharacter(FPrimitive):
    length: str

    def asC(self) -> str:
        return f"char*"


@dataclass
class Param:
    name: str
    op_arg: Optional[OP.Arg] = None

    # Param is not written directly or in any called subs
    is_const: Optional[bool] = None

    # Param is written locally
    is_const_local: Optional[bool] = None

    # Pairs of sub name and arg index where this param is used as an argument
    as_arg: Set[Tuple[str, int]] = field(default_factory=set)


@dataclass
class SubprogramInfo:
    name: str
    ast: f2003.Base

    params: List[Param] = field(default_factory=list)
    func_return_var: Optional[str] = None

    types: Dict[str, FType] = field(default_factory=dict)

    def isSubroutine(self) -> bool:
        return not self.is_function()

    def isFunction(self) -> bool:
        return self.func_return_var is not None

    def functionType(self) -> FType:
        assert(self.isFunction())
        return self.types[self.func_return_var]

    def paramNames(self) -> List[str]:
        return list(map(lambda p: p.name, self.params))

    def lookupParam(self, name: str) -> Optional[Param]:
        return safeFind(self.params, lambda arg: arg.name == name)


@dataclass
class Info:
    loop: OP.Loop
    config: Dict[str, Any]

    consts: Dict[str, FType] = field(default_factory=dict)

    entry_subprogram: Optional[str] = None
    subprograms: Dict[str, SubprogramInfo] = field(default_factory=dict)

    def functionNames(self) -> List[str]:
        return [s.name for s in self.subprograms.values() if s.isFunction()]


@dataclass
class Context:
    info: Info
    sub_info: SubprogramInfo

    def isEntry(self) -> bool:
        return self.sub_info.name == self.info.entry_subprogram

    def lookupType(self, name: str) -> Optional[FType]:
        return self.sub_info.types.get(name) or self.info.consts.get(name)

    def error(self, msg: str, node: Optional[Any] = None):
        full_msg = f"Error translating {self.sub_info.name}: {msg}"

        item = None
        if node is not None:
            item = fu.getItem(node)

        if item is not None:
            full_msg += f"\n  > {item.line}"

        raise OpError(full_msg)


def parseInfo(entities: List[Entity], app: Application, loop: OP.Loop, config: Dict[str, Any], const_rename: Optional[Callable[[str], str]] = None) -> Info:
    info = Info(loop, config)

    is_first = True
    for entity in entities:
        sub_info = None

        if isinstance(entity.ast, f2003.Subroutine_Subprogram):
            sub_info = parseSubroutineInfo(entity)
        elif isinstance(entity.ast, f2003.Function_Subprogram):
            sub_info = parseFunctionInfo(entity)
        else:
            raise OpError(f"Unknown top level entity AST type: {type(entity.ast)}")

        info.subprograms[sub_info.name] = sub_info

        if is_first:
            info.entry_subprogram = sub_info.name
            is_first = False

    if app.consts_module is not None:
        consts = parseConstsModule(app.consts_module, info)

        for name, type_ in consts.items():
            if isinstance(type_, FCharacter) or (isinstance(type_, FArray) and isinstance(type_.inner, FCharacter)):
                loop.consts.discard(name)

        for name, type_ in consts.items():
            actual_name = name
            if const_rename:
                actual_name = const_rename(name)

            info.consts[actual_name] = type_

    for sub_info in info.subprograms.values():
        if isinstance(sub_info.ast, f2003.Subroutine_Subprogram):
            parseSubroutineTypeInfo(Context(info, sub_info))
        elif isinstance(sub_info.ast, f2003.Function_Subprogram):
            parseFunctionTypeInfo(Context(info, sub_info))

    resolveOpArgs(entities, info, loop)
    resolveParamAccesses(info)

    return info


def parseConstsModule(program: Program, info: Info) -> Dict[str, FType]:
    module = fpu.get_child(program.ast, f2003.Module)

    if module is None:
        raise OpError("Invalid consts module")

    consts_info = SubprogramInfo("consts", None)

    spec_part = fpu.get_child(module, f2003.Specification_Part)
    return parseTypes(spec_part, Context(info, consts_info))


def parseSubroutineInfo(entity: Entity) -> SubprogramInfo:
    subroutine_stmt = fpu.get_child(entity.ast, f2003.Subroutine_Stmt)
    spec_part = fpu.get_child(entity.ast, f2003.Specification_Part)

    name = translateName(fpu.get_child(subroutine_stmt, f2003.Name))
    sub_info = SubprogramInfo(name, entity.ast)

    param_names = fpu.walk(fpu.get_child(subroutine_stmt, f2003.Dummy_Arg_List), f2003.Name)
    sub_info.params = [Param(translateName(name)) for name in param_names]

    return sub_info


def parseFunctionInfo(entity: Entity) -> SubprogramInfo:
    function_stmt = fpu.get_child(entity.ast, f2003.Function_Stmt)
    spec_part = fpu.get_child(entity.ast, f2003.Specification_Part)

    name = translateName(fpu.get_child(function_stmt, f2003.Name))
    sub_info = SubprogramInfo(name, entity.ast)

    param_list = fpu.get_child(function_stmt, f2003.Dummy_Arg_List)
    if param_list is not None:
        sub_info.params = [Param(translateName(name)) for name in fpu.walk(param_list, f2003.Name)]

    sub_info.func_return_var = f"_{sub_info.name}"

    suffix = fpu.get_child(function_stmt, f2003.Suffix)
    if suffix is not None:
        result_name = fpu.get_child(suffix, f2003.Result_Name)
        if result_name is not None:
            sub_info.func_return_var = f"_{translateName(result_name)}"

    return sub_info


def parseSubroutineTypeInfo(ctx: Context) -> None:
    spec_part = fpu.get_child(ctx.sub_info.ast, f2003.Specification_Part)
    ctx.sub_info.types = parseTypes(spec_part, ctx)


def parseFunctionTypeInfo(ctx: Context) -> None:
    spec_part = fpu.get_child(ctx.sub_info.ast, f2003.Specification_Part)
    ctx.sub_info.types = parseTypes(spec_part, ctx)

    if ctx.sub_info.func_return_var[1:] in ctx.sub_info.types:
        ctx.sub_info.types[ctx.sub_info.func_return_var] = ctx.sub_info.pop(ctx.sub_info.func_return_var[1:])

    function_stmt = fpu.get_child(ctx.sub_info.ast, f2003.Function_Stmt)
    prefix = fpu.get_child(function_stmt, f2003.Prefix)

    if prefix is not None:
        declaration_type_spec = fpu.get_child(prefix, f2003.Declaration_Type_Spec)
        derived_type_spec = fpu.get_child(prefix, f2003.Derived_Type_Spec)

        if declaration_type_spec is not None or derived_type_spec is not None:
            ctx.error("Unsupported non-intrinsic function return type", prefix)

        intrinsic_type_spec = fpu.get_child(prefix, f2003.Intrinsic_Type_Spec)
        if intrinsic_type_spec is not None:
            if ctx.sub_info.func_return_var in ctx.sub_info.types:
                ctx.error("Unexpected duplicate fuction type spec", intrinsic_type_spec)

            ctx.sub_info.types[ctx.sub_info.func_return_var] = parseIntrinsicType(intrinsic_type_spec, ctx)

    if ctx.sub_info.func_return_var not in ctx.sub_info.types:
        ctx.error("Could not resolve function type")

    if not isinstance(ctx.sub_info.types[ctx.sub_info.func_return_var], FPrimitive):
        ctx.error(f"Non-primitive function return: {func_type}")


def parseTypes(spec_part: f2003.Specification_Part, ctx: Context) -> Dict[str, FType]:
    type_map = {}

    for type_decl in fpu.walk(spec_part, f2003.Type_Declaration_Stmt):
        intrinsic_type_spec = fpu.get_child(type_decl, f2003.Intrinsic_Type_Spec)
        intrinsic_type = parseIntrinsicType(intrinsic_type_spec, ctx)

        attr_spec_list = type_decl.items[1]
        dimension_attr_spec = None
        attr_array_spec = None

        if attr_spec_list is not None:
            dimension_attr_spec = fpu.get_child(attr_spec_list, f2003.Dimension_Attr_Spec)

        if dimension_attr_spec is not None:
            attr_array_spec = parseArraySpec(dimension_attr_spec.items[1], ctx)

        entity_decl_list = fpu.get_child(type_decl, f2003.Entity_Decl_List)
        for entity_decl in entity_decl_list.items:
            name = translateName(fpu.get_child(entity_decl, f2003.Name), ctx)
            array_spec = entity_decl.items[1]

            if attr_array_spec is not None and array_spec is not None:
                ctx.error("Type declaration has both dimension() attr and array spec", type_decl)

            if array_spec is None:
                if attr_array_spec is not None:
                    type_map[name] = FArray(attr_array_spec, intrinsic_type)
                else:
                    type_map[name] = intrinsic_type
            else:
                type_map[name] = FArray(parseArraySpec(array_spec, ctx), intrinsic_type)

    return type_map


def parseIntrinsicType(intrinsic_type_spec: f2003.Intrinsic_Type_Spec, ctx: Context) -> FType:
    type_name = intrinsic_type_spec.items[0]

    kind = None
    if intrinsic_type_spec.items[1] is not None:
        kind_node = fpu.get_child(intrinsic_type_spec.items[1], f2003.Name)

        if kind_node is not None:
            kind = kind_node.string.upper()

    if type_name == "INTEGER":
        if kind in [None, "4", "IK", "IK4"]:
            return FInteger(4)
        elif kind in ["8", "IK8"]:
            return FInteger(8)

    elif type_name == "REAL":
        if kind in [None, "4", "RK4"]:
            return FReal(4)
        elif kind in ["8", "RK", "RK8"]:
            return FReal(8)

    elif type_name ==  "LOGICAL":
        if kind in [None, "LK"]:
            return FLogical()

    elif type_name == "CHARACTER":
        if isinstance(intrinsic_type_spec.items[1], f2003.Char_Selector):
            char_selector = intrinsic_type_spec.items[1]
            return FCharacter(translateGeneric(char_selector.items[0], ctx))
        elif isinstance(intrinsic_type_spec.items[1], f2003.Length_Selector):
            length_selector = intrinsic_type_spec.items[1]
            return FCharacter(translateGeneric(length_selector.items[1], ctx))
        else:
            ctx.error(f"Unknown character type spec", intrinsic_type_spec)

    ctx.error(f"Unable to parse intrinsic type", intrinsic_type_spec)


def parseArraySpec(array_spec: f2003.Array_Spec, ctx: Context) -> List[Tuple[str, str]]:
    if isinstance(array_spec, f2003.Explicit_Shape_Spec_List):
        shape_spec_list = []
        for shape_spec in array_spec.items:
            lb = shape_spec.items[0]
            ub = shape_spec.items[1]

            if lb is None:
                shape_spec_list.append(("1", translateGeneric(ub, ctx)))
            else:
                shape_spec_list.append((translateGeneric(lb, ctx), translateGeneric(ub, ctx)))

        return shape_spec_list

    ctx.error(f"Unsupported array spec", array_spec)


def setOpArg(entity: Entity, param_idx: int, info: Info, op_arg: OP.Arg) -> None:
    def_stmt = fpu.get_child(entity.ast, (f2003.Subroutine_Stmt, f2003.Function_Stmt))
    name = translateName(fpu.get_child(def_stmt, f2003.Name))

    sub_info = info.subprograms[name]
    sub_info.params[param_idx].op_arg = op_arg


def resolveOpArgs(entities: List[Entity], info: Info, loop: OP.Loop) -> None:
    for i in range(len(loop.args)):
        fu.mapParam(entities[0], i, entities, setOpArg, info, loop.args[i])


def resolveParamAccesses(info: Info) -> None:
    for sub_info in info.subprograms.values():
        resolveParamAccessesLocal(Context(info, sub_info))

    has_unresolved = True

    while has_unresolved:
        has_unresolved = False

        for sub_info in info.subprograms.values():
            unresolved = tryResolveParams(Context(info, sub_info))
            has_unresolved = has_unresolved or unresolved


def resolveParamAccessesLocal(ctx: Context) -> None:
    execution_part = fpu.get_child(ctx.sub_info.ast, f2003.Execution_Part)
    names = fpu.walk(execution_part, f2003.Name)

    assignments = fpu.walk(execution_part, f2003.Assignment_Stmt)
    loop_controls = fpu.walk(execution_part, f2003.Loop_Control)

    assigned_to = set()
    for assignment in assignments:
        lhs = assignment.items[0]

        if isinstance(lhs, f2003.Name):
            assigned_to.add(translateName(lhs, ctx))
        elif isinstance(lhs, f2003.Part_Ref):
            name = translateName(fpu.get_child(lhs, f2003.Name), ctx)
            assigned_to.add(name)
        else:
            ctx.error(f"Unknown assignment LHS", assignment)

    for loop_control in loop_controls:
        if loop_control.items[0] != None:
            continue

        if not isinstance(loop_control.items[1][0], f2003.Name):
            ctx.error(f"Unknown loop control", loop_control)

        control_var = translateName(loop_control.items[1][0], ctx)
        assigned_to.add(control_var)

    for param in ctx.sub_info.params:
        param.is_const_local = not (param.name in assigned_to)

        param_refs = filter(lambda n: translateName(n, ctx) == param.name, names)
        for ref in param_refs:
            call = getCall(ref, ctx)

            if call is not None:
                param.as_arg.add(call)

        if param.is_const_local and len(param.as_arg) == 0:
            param.is_const = True
        elif not param.is_const_local:
            param.is_const = False

        if param.is_const == None and ("atomicAdd", 0) in param.as_arg:
            param.is_const = False


def tryResolveParams(ctx: Context) -> bool:
    has_unresolved = False

    for param in ctx.sub_info.params:
        if param.is_const is not None:
            continue

        all_const = True
        unresolved = False

        for target_name, idx in param.as_arg:
            target_sub = ctx.info.subprograms[target_name]

            if target_sub.params[idx].is_const is None:
                unresolved = True
                continue

            if target_sub.params[idx].is_const == False:
                all_const = False
                break

        if not all_const:
            param.is_const = False
        elif all_const and not unresolved:
            param.is_const = True
        else:
            has_unresolved = True

    return has_unresolved


def getCall(ref: f2003.Name, ctx: Context) -> Optional[Tuple[str, int]]:
    node = ref
    parent = getattr(node, "parent", None)

    if isinstance(parent, f2003.Part_Ref):
        if not hasattr(parent, "parent"):
            return None

        node = parent
        parent = node.parent

    if parent is None:
        return None

    if isinstance(parent, (f2003.Actual_Arg_Spec_List, f2003.Section_Subscript_List)):
        func_name_node = fpu.get_child(parent.parent, f2003.Name)

        if func_name_node is None:
            return None

        func_name = translateName(func_name_node, ctx)
        sub_info = ctx.info.subprograms.get(func_name)

        if func_name != "atomicAdd" and sub_info is None:
            return None

        arg_idx = [id(item) for item in parent.items].index(id(node))
        return (func_name, arg_idx)

    return None


def translate(info: Info) -> str:
    decls = ""
    srcs = ""

    for sub_info in info.subprograms.values():
        decl, src = translateSubprogram(Context(info, sub_info))

        decls += decl + "\n\n"
        srcs += src + "\n\n"

    return decls + "\n" + srcs


def translateSubprogram(ctx: Context) -> Tuple[str, str]:
    spec_part = fpu.get_child(ctx.sub_info.ast, f2003.Specification_Part)
    execution_part = fpu.get_child(ctx.sub_info.ast, f2003.Execution_Part)

    param_decls = []

    for param in ctx.sub_info.params:
        assert(param.is_const is not None)
        param_type = ctx.sub_info.types[param.name]

        if isinstance(param_type, FPrimitive):
            if param.is_const and not (hasattr(param.op_arg, "opt") and param.op_arg.opt):
                param_decls.append(f"const {param_type.asC()} {param.name}")
            elif param.is_const:
                param_decls.append(f"const {param_type.asC()}& {param.name}")
            else:
                param_decls.append(f"{param_type.asC()}& {param.name}")
        else:
            const = ""
            if param.is_const:
                const = "const "

            param_decls.append(f"f2c::Ptr<{const}{param_type.inner.asC()}> _f2c_ptr_{param.name}")

    return_type = "void"

    if ctx.sub_info.isFunction():
        return_type = ctx.sub_info.functionType().asC()

    prefix = ctx.info.config.get("func_prefix")

    if prefix:
        prefix = f"{prefix} "
    else:
        prefix = ""

    src_decl  = f"static {prefix}{return_type} {ctx.sub_info.name}(\n    "
    src_decl += ",\n    ".join(param_decls) + "\n)"

    src = src_decl + " "
    src += "{\n"
    src += indent(translateSpecificationPart(spec_part, ctx)) + "\n"
    src += indent(translateExecutionPart(execution_part, ctx))
    src += "\n}"

    return src_decl + ";", src


def translateSpecificationPart(spec_part: f2003.Specification_Part, ctx: Context) -> str:
    init_src = ""
    parameters = {}

    has_implicit_none = False

    for node in spec_part.content:
        if isinstance(node, f2003.Use_Stmt):
            continue

        if isinstance(node, f2003.Implicit_Part):
            for implicit_node in node.children:
                if isinstance(implicit_node, f2003.Implicit_Stmt) and implicit_node.items[0] == "NONE":
                    has_implicit_none = True
                elif isinstance(implicit_node, f2003.Parameter_Stmt):
                    def_list = implicit_node.items[1]

                    for named_constant_def in def_list.children:
                        name = translateGeneric(named_constant_def.items[0], ctx)
                        val  = translateGeneric(named_constant_def.items[1], ctx)

                        parameters[name] = val
                else:
                    ctx.error(f"Unsupported implicit part statement", node)

            continue

        if isinstance(node, f2003.Type_Declaration_Stmt):
            continue

        if isinstance(node, f2003.Data_Stmt):
            init_src += translateDataStmt(node, ctx)
            continue

        ctx.error(f"Unsupported specification statement", node)

    if not has_implicit_none:
        ctx.error(f"No implicit none", spec_part)

    src = ""
    for name, type_ in ctx.sub_info.types.items():
        if name in ctx.sub_info.paramNames() and isinstance(type_, FArray):
            src += f"const {type_.asSpan('_f2c_ptr_' + name, name, ctx)};\n"
            continue

        if name in ctx.sub_info.paramNames():
            continue

        # TODO: probably not correct, might need check for external?
        if name in ctx.info.functionNames():
            continue

        if name in parameters:
            assert(not isinstance(type_, FArray))
            src += f"constexpr {type_.asLocal(name)} = {parameters[name]};\n"
        elif isinstance(type_, FPrimitive):
            src += f"{type_.asLocal(name)};\n"
        else:
            src += f"{type_.asLocal('_f2c_arr_' + name)};\n"
            src += f"const {type_.asSpan('f2c::Ptr{_f2c_arr_' + name + '}', name, ctx)};\n"

    return src + "\n" + init_src


def translateDataStmt(data_stmt: f2003.Data_Stmt, ctx: Context) -> str:
    src = "";
    for child in data_stmt.items:
        assert(isinstance(child, f2003.Data_Stmt_Set))

        object_list = child.items[0]
        value_list = child.items[1]

        if len(value_list.children) != 1:
            ctx.error(f"Unsupported multiple value data statement", child)

        value = value_list.children[0]
        if value.items[1] is not None:
            ctx.error(f"Unsupported repeat in data statement", child)

        value = translateGeneric(value, ctx)

        assert(len(object_list.children) > 0)
        for data_object in object_list.children:
            src += f"{translateGeneric(data_object, ctx)} = {value};\n"

    return src


def translateExecutionPart(execution_part: f2003.Execution_Part, ctx: Context) -> str:
    src = ""
    last = ""

    for stmt in execution_part.content:
        last = translateGeneric(stmt, ctx)
        src += last

    if ctx.sub_info.isFunction() and not last.startswith("return"):
        src += f"return {ctx.sub_info.func_return_var};\n"

    return src


def translateAssignmentStmt(assignment_stmt: f2003.Assignment_Stmt, ctx: Context) -> str:
    lhs = translateGeneric(assignment_stmt.items[0], ctx)
    return f"{lhs} = {translateGeneric(assignment_stmt.items[2], ctx)};\n"


def translateCallStmt(call_stmt: f2003.Call_Stmt, ctx: Context) -> str:
    call_target = translateGeneric(call_stmt.items[0], ctx)
    if call_stmt.items[1] is None:
        return f"{call_target}();\n"
    else:
        actual_arg_spec_list = call_stmt.items[1]
        return f"{call_target}({translateArgList(list(actual_arg_spec_list.items), ctx, call_target)});\n"


def translateContinueStmt(continue_stmt: f2003.Continue_Stmt, ctx: Context) -> str:
    return f"// {continue_stmt}\n"


def translateIfStmt(if_stmt: f2003.If_Stmt, ctx: Context) -> str:
    return f"if ({translateGeneric(if_stmt.items[0], ctx)}) {{\n{indent(translateGeneric(if_stmt.items[1], ctx))}\n}}\n"


def translateReturnStmt(return_stmt: f2003.Return_Stmt, ctx: Context) -> str:
    if return_stmt.items[0] is not None:
        ctx.error("Labelled return not supported", return_stmt)

    if ctx.sub_info.isFunction():
        return f"return {ctx.sub_info.func_return_var};\n"

    return f"return;\n"


def translateStopStmt(stop_stmt: f2003.Stop_Stmt, ctx: Context) -> str:
    src = "assert(false);\n"
    return src


def translateWriteStmt(write_stmt: f2003.Write_Stmt, ctx: Context) -> str:
    return f"// {write_stmt}\n"


def translateBlockNonlabelDoConstruct(block_nonlabel_do_construct: f2003.Block_Nonlabel_Do_Construct, ctx: Context) -> str:
    src = ""

    for child in block_nonlabel_do_construct.content:
        if isinstance(child, f2003.Nonlabel_Do_Stmt):
            if child.item.name != None:
                ctx.error("Unsupported labelled do construct", child)

            loop_control = child.items[1]

            # While loop
            if loop_control.items[0] != None:
                src += f"while ({translateGeneric(loop_control.items[0], ctx)})" + " {\n"
                continue

            control_var = translateGeneric(loop_control.items[1][0], ctx)
            bounds = loop_control.items[1][1]

            lb = translateGeneric(bounds[0], ctx)
            ub = translateGeneric(bounds[1], ctx)

            if len(bounds) == 2:
                src += f"for ({control_var} = {lb}; {control_var} <= {ub}; ++{control_var})" + " {\n"
            else:
                step = translateGeneric(bounds[2], ctx)
                src += f"for ({control_var} = {lb}; {control_var} <= {ub}; {control_var} += {step})" + " {\n"

        elif isinstance(child, f2003.End_Do_Stmt):
            src += "}\n"
        else:
            src += f"{indent(translateGeneric(child, ctx))}\n"

    return src


def translateIfConstruct(if_construct: f2003.If_Construct, ctx: Context) -> str:
    src = ""

    for child in if_construct.content:
        if isinstance(child, f2003.If_Then_Stmt):
            src += f"if ({translateGeneric(child.items[0], ctx)}) {{\n"
        elif isinstance(child, f2003.Else_If_Stmt):
            src += f"}} else if ({translateGeneric(child.items[0], ctx)}) {{\n"
        elif isinstance(child, f2003.Else_Stmt):
            src += f"}} else {{\n"
        elif isinstance(child, f2003.End_If_Stmt):
            src += f"}}\n"
        else:
            src += f"{indent(translateGeneric(child, ctx))}\n"

    return src


def translateName(name: f2003.Name, ctx: Optional[Context] = None) -> str:
    raw = name.string.lower()

    parent = getattr(name, "parent", None)
    if parent is not None and isinstance(parent, f2003.Intrinsic_Function_Reference):
        return raw

    rename = {
        "atomicadd": "atomicAdd",
    }

    if raw in rename:
        return rename[raw]

    cxx_keywords = [
        "alignas",
        "alignof",
        "and",
        "and_eq",
        "asm",
        "atomic_cancel",
        "atomic_commit",
        "atomic_noexcept",
        "auto",
        "bitand",
        "bitor",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "class",
        "compl",
        "concept",
        "const",
        "consteval",
        "constexpr",
        "constinit",
        "const_cast",
        "continue",
        "co_await",
        "co_return",
        "co_yield",
        "decltype",
        "default",
        "delete",
        "do",
        "double",
        "dynamic_cast",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "not",
        "not_eq",
        "nullptr",
        "operator",
        "or",
        "or_eq",
        "private",
        "protected",
        "public",
        "reflexpr",
        "register",
        "reinterpret_cast",
        "requires",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "synchronized",
        "template",
        "this",
        "thread_local",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "wchar_t",
        "while",
        "xor",
        "xor_eq"
    ]

    if raw in cxx_keywords:
        raw = f"_op2k_{raw}"

    if ctx is not None and ctx.sub_info.isFunction() and raw == ctx.sub_info.func_return_var[1:]:
        return ctx.sub_info.func_return_var

    return raw


def translatePartRef(part_ref: f2003.Part_Ref, ctx: Context) -> str:
    name = translateName(fpu.get_child(part_ref, f2003.Name), ctx)
    subscript_list = fpu.get_child(part_ref, f2003.Section_Subscript_List)

    if subscript_list is None:
        ctx.error("Part-ref has no subscript list", part_ref)

    if name in ctx.info.functionNames():
        return f"{name}({translateArgList(subscript_list.children, ctx, name)})"

    if len(subscript_list.children) == 0:
        ctx.error("Unexpected zero-length part-ref subscript list", part_ref)

    array_type = ctx.lookupType(name)

    if array_type is None:
        ctx.error(f"Could not find type of part-ref", part_ref)

    if name in ctx.sub_info.types:
        is_slice = any(isinstance(child, f2003.Subscript_Triplet) for child in subscript_list.children)

        if not is_slice:
            return f"{name}({', '.join(translateGeneric(child, ctx) for child in subscript_list.children)})"


        if len(subscript_list.children) != len(array_type.shape):
            ctx.error(f"Number of subscripts doesn't match array type {array_type}", subscript_list)

        extents = []
        for shape, child in zip(array_type.shape, subscript_list.children):
            if not isinstance(child, f2003.Subscript_Triplet):
                idx = translateGeneric(child, ctx)
                extents.append(f"f2c::Extent{{{idx}, {idx}}}")
                continue

            if child.items[2] is not None:
                ctx.error(f"Unsupported stride in subscript triplet", child)

            lb = ""
            if child.items[0] is not None:
                lb = translateGeneric(child.items[0], ctx)
            else:
                lb = shape[0]

            ub = ""
            if child.items[1] is not None:
                ub = translateGeneric(child.items[1], ctx)
            else:
                ub = shape[1]

            extents.append(f"f2c::Extent{{{lb}, {ub}}}")

        return f"{name}.slice({', '.join(extents)})"

    # Only thing left should be array consts
    sizes = []
    for lb, ub in array_type.shape:
        if lb == "1":
            sizes.append(f"({ub})")
        else:
            sizes.append(f"(1 + {ub} - ({lb}))")

    if array_type.shape[0][0] == "1":
        index = f"{translateGeneric(subscript_list.children[0], ctx)}"
    else:
        index = f"({translateGeneric(subscript_list.children[0], ctx)} + 1 - ({array_type.shape[0][0]}))"

    for i, extra_index in enumerate(subscript_list.children[1:], start=1):
        index += f" + ({translateGeneric(extra_index, ctx)} - ({array_type.shape[i][0]})) * {'*'.join(sizes[:i])}"

    return f"{name}[({index}) - 1]"


def translateIntrinsicFunctionReference(intrinsic_function_reference: f2003.Intrinsic_Function_Reference, ctx: Context) -> str:
    intrinsic_funcs = {
        "abs":   "f2c::abs",
        "dble":  "f2c::dble",
        "int":   "f2c::int_",
        "min":   "f2c::min",
        "max":   "f2c::max",
        "mod":   "f2c::mod",
        "nint":  "f2c::nint",
        "sign":  "f2c::copysign",

        "acos":  "f2c::acos",
        "asin":  "f2c::asin",
        "atan":  "f2c::atan",
        "atan2": "f2c::atan2",
        "cos":   "f2c::cos",
        "cosh":  "f2c::cosh",
        "exp":   "f2c::exp",
        "log":   "f2c::log",
        "log10": "f2c::log10",
        "sin":   "f2c::sin",
        "sinh":  "f2c::sinh",
        "sqrt":  "f2c::sqrt",
        "tan":   "f2c::tan",
        "tanh":  "f2c::tanh",

        "dabs":  "f2c::abs",
        "dacos": "f2c::acos",
        "dasin": "f2c::asin",
        "datan": "f2c::atan",
        "dcos":  "f2c::cos",
        "dcosh": "f2c::cosh",
        "dexp":  "f2c::exp",
        "dint":  "f2c::int",
        "dsign": "f2c::copysign",
        "dsin":  "f2c::sin",
        "dsinh": "f2c::sinh",
        "dsqrt": "f2c::sqrt",
        "dtan":  "f2c::tan",
        "dtanh": "f2c::tanh",
    }

    items = intrinsic_function_reference.items
    func_name = translateName(items[0], ctx)

    if func_name not in intrinsic_funcs:
        ctx.error(f"Unsupported intrinsic func: {func_name}", intrinsic_function_reference)

    return f"{intrinsic_funcs[func_name]}({translateGeneric(items[1], ctx)})"


def translateActualArgSpecList(actual_arg_spec_list: f2003.Actual_Arg_Spec_List, ctx: Context) -> str:
    return ", ".join([translateGeneric(item, ctx) for item in actual_arg_spec_list.items])


# Handles Actual_Arg_Spec_Lists and Section_Subscript_Lists for functions
def translateArgList(arg_list: List[f2003.Base], ctx: Context, call_target: str) -> str:
    if call_target == "atomicAdd":
        assert(len(arg_list) == 2)
        return ", ".join([f"&({translateGeneric(arg_list[0], ctx)})", translateGeneric(arg_list[1], ctx)])

    target_sub = ctx.info.subprograms[call_target]

    args = []
    for item, target_type in zip(arg_list, [target_sub.types[param.name] for param in target_sub.params]):
        arg = None

        if isinstance(target_type, FArray) and isinstance(item, f2003.Part_Ref):
            name = translateName(fpu.get_child(item, f2003.Name), ctx)
            if name in ctx.info.functionNames():
                ctx.error(f"Unsupported function return as array argument", item)

            if name not in ctx.sub_info.types:
                ctx.error(f"Unsupported const part-ref as arg", item)

            sl = fpu.get_child(item, f2003.Section_Subscript_List)
            if any(isinstance(c, f2003.Subscript_Triplet) for c in sl.children):
                ctx.error(f"Unsupported slice part-ref as arg")

            arg = f"{name}.ptr_at({', '.join(translateGeneric(c, ctx) for c in sl.children)})"
        elif isinstance(target_type, FArray) and not isinstance(item, f2003.Name):
            ctx.error(f"Unsupported expression passed to array argument", item)
        else:
            arg = translateGeneric(item, ctx)

        args.append(arg)

    return ", ".join(args)


def translateParenthesis(parenthesis: f2003.Parenthesis, ctx: Context) -> str:
    return f"({translateGeneric(parenthesis.items[1], ctx)})"


def translateMultOperand(mult_operand: f2003.Mult_Operand, ctx: Context) -> str:
    items = mult_operand.items
    return f"f2c::pow({translateGeneric(items[0], ctx)}, {translateGeneric(items[2], ctx)})"


def translateAddOperand(add_operand: f2003.Add_Operand, ctx: Context) -> str:
    items = add_operand.items
    return f"{translateGeneric(items[0], ctx)} {items[1]} {translateGeneric(items[2], ctx)}"


def translateLevel2Expr(level_2_expr: f2003.Level_2_Expr, ctx: Context) -> str:
    items = level_2_expr.items
    return f"{translateGeneric(items[0], ctx)} {items[1]} {translateGeneric(items[2], ctx)}"


def translateLevel2UnaryExpr(level_2_unary_expr: f2003.Level_2_Unary_Expr, ctx: Context) -> str:
    items = level_2_unary_expr.items
    return f"{items[0]}{translateGeneric(items[1], ctx)}"


def translateLevel4Expr(level_4_expr: f2003.Level_4_Expr, ctx: Context) -> str:
    items = level_4_expr.items
    op = items[1]

    op_map = {
        ".EQ.": "==",
        ".NE.": "!=",
        ".LT.": "<",
        ".LE.": "<=",
        ".GT.": ">",
        ".GE.": ">="
    }

    if op in op_map:
        op = op_map[op]

    return f"{translateGeneric(items[0], ctx)} {op} {translateGeneric(items[2], ctx)}"


def translateAndOperand(and_operand: f2003.And_Operand, ctx: Context) -> str:
    return f"!{translateGeneric(and_operand.items[1], ctx)}"


def translateOrOperand(or_operand: f2003.Or_Operand, ctx: Context) -> str:
    return f"{translateGeneric(or_operand.items[0], ctx)} && {translateGeneric(or_operand.items[2], ctx)}"


def translateEquivOperand(equiv_operand: f2003.Equiv_Operand, ctx: Context) -> str:
    return f"{translateGeneric(equiv_operand.items[0], ctx)} || {translateGeneric(equiv_operand.items[2], ctx)}"


# def translateExpr(expr: f2003.Expr, ctx: Context) -> str:
#     return str(expr)


def translateIntLiteralConstant(int_literal_constant: f2003.Int_Literal_Constant, ctx: Context) -> str:
    if int_literal_constant.items[1] is not None:
        ctx.error(f"Unsupported int literal kind specifier: {int_literal_constant.items[1]}", int_literal_constant)

    return str(int_literal_constant.items[0])


def translateRealLiteralConstant(real_literal_constant: f2003.Real_Literal_Constant, ctx: Context) -> str:
    is_float = False

    kind_spec = real_literal_constant.items[1]
    kind_spec_is_float = {
        None: True,
        "RK": False,
        "RK8": False,
        "RK4": True
    }

    if kind_spec not in kind_spec_is_float:
        ctx.error(f"Unsupported real literal kind specifier: {kind_spec}", real_literal_constant)

    is_float = kind_spec_is_float[kind_spec]
    raw = str(real_literal_constant.items[0])

    if "E" in raw:
        is_float = True
        raw = raw.replace("E", "e")
    elif "D" in raw:
        is_float = False
        raw = raw.replace("D", "e")

    if is_float:
        return raw + "f"

    return raw


def translateLogicalLiteralConstant(logical_literal_constant: f2003.Logical_Literal_Constant, ctx: Context) -> str:
    if logical_literal_constant.items[0] == ".TRUE.":
        return "true"
    else:
        return "false"


def translateCharLiteralConstant(char_literal_constant: f2003.Char_Literal_Constant, ctx: Context) -> str:
    return char_literal_constant.items[0].replace("'", '"')


TRANSLATE_TABLE = {
    # Action_Stmt Executable_Constructs

    # f2003.Allocate_Stmt
    f2003.Assignment_Stmt: translateAssignmentStmt,
    # f2003.Backspace_Stmt
    f2003.Call_Stmt: translateCallStmt,
    # f2003.Close_Stmt
    f2003.Continue_Stmt: translateContinueStmt,
    # f2003.Cycle_Stmt
    # f2003.Deallocate_Stmt
    # f2003.Endfile_Stmt
    # f2003.End_Function_Stmt
    # f2003.End_Program_Stmt
    # f2003.End_Subroutine_Stmt
    # f2003.Exit_Stmt
    # f2003.Flush_Stmt
    # f2003.Forall_Stmt
    # f2003.Goto_Stmt
    f2003.If_Stmt: translateIfStmt,
    f2008.If_Stmt: translateIfStmt,
    # f2003.Inquire_Stmt
    # f2003.Nullify_Stmt
    # f2003.Open_Stmt
    # f2003.Pointer_Assignment_Stmt
    # f2003.Print_Stmt
    # f2003.Read_Stmt
    f2003.Return_Stmt: translateReturnStmt,
    # f2003.Rewind_Stmt
    f2003.Stop_Stmt: translateStopStmt,
    # f2003.Wait_Stmt
    # f2003.Where_Stmt
    f2003.Write_Stmt: translateWriteStmt,
    # f2003.Arithmetic_If_Stmt
    # f2003.Computed_Goto_Stmt


    # Other Executable_Constructs

    # f2003.Associate_Construct
    # f2003.Case_Construct
    # f2003.Block_Do_Construct
    f2003.Block_Nonlabel_Do_Construct: translateBlockNonlabelDoConstruct,
    f2008.Block_Nonlabel_Do_Construct: translateBlockNonlabelDoConstruct,
    # f2003.Forall_Construct
    f2003.If_Construct: translateIfConstruct,
    # f2003.Select_Type_Construct
    # f2003.Where_Construct


    # Misc

    f2003.Name: translateName,
    f2003.Part_Ref: translatePartRef,

    f2003.Intrinsic_Function_Reference: translateIntrinsicFunctionReference,
    f2003.Actual_Arg_Spec_List: translateActualArgSpecList,


    # Expr stuff

    f2003.Parenthesis: translateParenthesis,
    # f2003.Level_1_Expr
    # f2003.Defined_Unary_Op
    # f2003.Defined_Op
    f2003.Mult_Operand: translateMultOperand,
    f2003.Add_Operand: translateAddOperand,
    f2003.Level_2_Expr: translateLevel2Expr,
    f2003.Level_2_Unary_Expr: translateLevel2UnaryExpr,
    # f2003.Level_3_Expr
    f2003.Level_4_Expr: translateLevel4Expr,
    f2003.And_Operand: translateAndOperand,
    f2003.Or_Operand: translateOrOperand,
    f2003.Equiv_Operand: translateEquivOperand,
    # f2003.Level_5_Expr

    # f2003.Expr: translateExpr,


    # Literal constants
    f2003.Int_Literal_Constant: translateIntLiteralConstant,
    f2003.Real_Literal_Constant: translateRealLiteralConstant,
    # f2003.Complex_Literal_Constant
    f2003.Logical_Literal_Constant: translateLogicalLiteralConstant,
    f2003.Char_Literal_Constant: translateCharLiteralConstant,
    # f2003.Boz_Literal_Constant
}


def translateGeneric(node: f2003.Base, ctx: Context) -> str:
    for type_ in TRANSLATE_TABLE.keys():
        if type(node) == type_:
            return TRANSLATE_TABLE[type_](node, ctx)

    ctx.error(f"Type {type(node)} not registered for translateGeneric ({node})", node)

