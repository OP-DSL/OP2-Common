"""
Fortran parser backend driven by LLVM Flang (via the op2-flang-scan helper binary).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import op as OP
from store import Application, Function, Location, ParseError, Program

# re-use the type string parser from the fparser2 path, so both paths
# produce identical OP.Type instances for the same input
from fortran.parser import parseType  # noqa: E402


# Binary discovery

DEFAULT_BIN_NAME = "op2-flang-scan"


def _repo_build_candidates() -> List[Path]:
    """
    Default locations of op2-flang-scan relative to this file.

    The library build (`make -C op2`) installs the binary into `op2/bin`.
    A standalone CMake build lands in `translator-v2/flang-scan/build`.
    """
    here = Path(__file__).resolve()
    translator_v2 = here.parents[2]
    repo_root = translator_v2.parent
    return [
        repo_root / "op2" / "bin" / DEFAULT_BIN_NAME,
        translator_v2 / "flang-scan" / "build" / DEFAULT_BIN_NAME,
    ]


def _repo_build_candidate() -> Optional[Path]:
    """
    Return the first existing default build location of op2-flang-scan.
    """
    for candidate in _repo_build_candidates():
        if candidate.is_file():
            return candidate
    return None


def resolve_scan_binary(override: Optional[str] = None) -> Path:
    """
    Locate the op2-flang-scan binary.

    Lookup order:
      1. explicit override (CLI flag)
      2. OP2_FLANG_SCAN environment variable
      3. op2/bin/op2-flang-scan (library build install)
      4. translator-v2/flang-scan/build/op2-flang-scan
      5. op2-flang-scan on PATH
    """
    if override:
        p = Path(override)
        if not p.is_file():
            raise ParseError(f"op2-flang-scan not found at --flang-scan path: {p}")
        return p

    env = os.environ.get("OP2_FLANG_SCAN")
    if env:
        p = Path(env)
        if not p.is_file():
            raise ParseError(f"op2-flang-scan not found at OP2_FLANG_SCAN: {p}")
        return p

    repo = _repo_build_candidate()
    if repo is not None:
        return repo

    found = shutil.which(DEFAULT_BIN_NAME)
    if found:
        return Path(found)

    raise ParseError(
        "op2-flang-scan binary not found. Build it with `make -C op2` when LLVM "
        "Flang is configured, from translator-v2/flang-scan, or set "
        "OP2_FLANG_SCAN / --flang-scan <path>."
    )


# Scanner invocation

def run_scan(
    source: str,
    path: Path,
    scan_bin: Path,
    include_dirs: Optional[Iterable[Path]] = None,
) -> Dict[str, Any]:
    """
    Run op2-flang-scan on the given (already preprocessed) source text and
    return the parsed JSON document.

    `include_dirs` are forwarded as `-I` so Flang can resolve Fortran
    `INCLUDE` the same way fparser2's `FortranStringReader` does.
    """
    results = run_scan_batch([(path, source)], scan_bin, include_dirs=include_dirs)
    data = results[path]
    if data.get("error"):
        raise ParseError(f"op2-flang-scan failed for {path}: {data['error']}")
    return data


def run_scan_batch(
    units: List[Tuple[Path, str]],
    scan_bin: Path,
    include_dirs: Optional[Iterable[Path]] = None,
) -> Dict[Path, Dict[str, Any]]:
    """
    Scan many preprocessed translation units using a single op2-flang-scan process.

    Each unit still gets its own Flang Prescan+Parse (Flang has no multi-TU parsing),
    but LLVM load / process spawn happens only once.

    Returns a map from input `Path` to the per-file JSON document.
    Failed units have an `"error"` string field instead of usable `events`.
    """
    if not units:
        return {}

    # single-unit fast path uses --stdin CLI (and cwd hint)
    if len(units) == 1:
        path, source = units[0]
        cmd = [str(scan_bin), "--stdin", "--path", str(path)]
        for d in include_dirs or []:
            cmd.extend(["-I", str(d)])
        try:
            proc = subprocess.run(
                cmd,
                input=source.encode("utf-8"),
                capture_output=True,
                check=False,
                cwd=str(path.parent) if path.parent else None,
            )
        except FileNotFoundError as e:
            raise ParseError(f"failed to launch op2-flang-scan: {e}")
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise ParseError(
                f"op2-flang-scan failed for {path} (exit {proc.returncode}):\n{stderr}"
            )
        try:
            return {path: json.loads(proc.stdout.decode("utf-8", errors="replace"))}
        except json.JSONDecodeError as e:
            raise ParseError(f"op2-flang-scan produced invalid JSON for {path}: {e}")

    cmd = [str(scan_bin), "--batch"]
    for d in include_dirs or []:
        cmd.extend(["-I", str(d)])

    # OP2_FLANG_BATCH_V1 framing (see op2-flang-scan.cpp runBatchMode)
    chunks: List[bytes] = [b"OP2_FLANG_BATCH_V1\n"]
    path_by_reported: Dict[str, Path] = {}
    for path, source in units:
        reported = str(path)
        path_by_reported[reported] = path
        body = source.encode("utf-8")
        chunks.append(reported.encode("utf-8") + b"\n")
        chunks.append(str(len(body)).encode("ascii") + b"\n")
        chunks.append(body)

    try:
        proc = subprocess.run(
            cmd,
            input=b"".join(chunks),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise ParseError(f"failed to launch op2-flang-scan: {e}")

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    results: Dict[Path, Dict[str, Any]] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as e:
            raise ParseError(
                f"op2-flang-scan --batch produced invalid JSON line: {e}\n{line[:200]}"
            )
        reported = str(doc.get("path", ""))
        key = path_by_reported.get(reported)
        if key is None:
            # fall back to matching by resolved path string
            for p, _ in units:
                if str(p) == reported:
                    key = p
                    break
        if key is None:
            raise ParseError(
                f"op2-flang-scan --batch returned unexpected path: {reported!r}"
            )
        results[key] = doc

    if len(results) != len(units):
        missing = [str(p) for p, _ in units if p not in results]
        raise ParseError(
            f"op2-flang-scan --batch missing results for: {', '.join(missing)}\n"
            f"stderr:\n{stderr}"
        )

    if proc.returncode != 0 and all(r.get("error") for r in results.values()):
        raise ParseError(
            f"op2-flang-scan --batch failed for all units "
            f"(exit {proc.returncode}):\n{stderr}"
        )

    return results


# JSON -> OP objects

def _arg_as_access_type(arg: Dict[str, Any], loc: Location) -> OP.AccessType:
    name = _arg_as_name(arg, loc).upper()
    mapping = {
        "OP_READ": 0, "OP_WRITE": 1, "OP_RW": 2,
        "OP_INC": 3, "OP_MIN": 4, "OP_MAX": 5, "OP_WORK": 6,
    }
    if name not in mapping:
        raise ParseError(
            f"invalid access type {name}, expected one of {', '.join(mapping)}", loc
        )
    return OP.AccessType(mapping[name])


def _arg_as_int(arg: Dict[str, Any], loc: Location, optional: bool = False) -> Optional[int]:
    kind = arg.get("kind")
    if kind == "int":
        return int(arg["value"])
    if kind == "raw":
        # try parsing the raw source text
        text = str(arg.get("source", "")).strip()
        try:
            return int(text)
        except ValueError:
            pass
    if optional:
        return None
    raise ParseError(f"expected integer literal, got {kind} ({arg!r})", loc)


def _arg_as_name(arg: Dict[str, Any], loc: Location) -> str:
    if arg.get("kind") != "name":
        raise ParseError(
            f"expected identifier, got {arg.get('kind')} ({arg!r})", loc
        )
    return str(arg["value"]).lower()


def _arg_as_string(arg: Dict[str, Any], loc: Location) -> str:
    kind = arg.get("kind")
    if kind == "string":
        return str(arg["value"])
    if kind == "raw":
        # if the C++ visitor could not recognise a character literal, but the raw
        # source text still looks like a quoted string, accept it
        text = str(arg.get("source", "")).strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            return text[1:-1]
    raise ParseError(
        f"expected character literal, got {kind} ({arg!r})", loc
    )


def _make_loc(path: Path, node_loc: Optional[Dict[str, int]]) -> Location:
    line = 0
    col = 0
    if node_loc is not None:
        line = int(node_loc.get("line", 0) or 0)
        col = int(node_loc.get("column", 0) or 0)
    return Location(str(path), line, col)


# Event dispatch

_PAR_LOOP_RE = re.compile(r"^op_par_loop_\d+$")
_SUBPROGRAM_KINDS = ("subroutine_subprogram", "function_subprogram")


def _flang_call_names(entity: Function) -> Set[str]:
    """
    Get the lower-cased names of unambiguous `CALL` statements in `flang_body`.
    """
    body = getattr(entity, "flang_body", None) or {}
    names: Set[str] = set()
    for call in body.get("calls") or []:
        name = str(call.get("name") or "").lower()
        if name:
            names.add(name)
    return names


def _function_from_subprogram_event(event: Dict[str, Any], program: Program) -> Function:
    """
    Create a Function entity from a subprogram event.
    """
    name = str(event.get("name", "")).lower()
    parameters = [str(p).lower() for p in event.get("parameters", [])]
    depends = {str(d).lower() for d in event.get("depends", [])}

    entity = Function(name, None, program)
    entity.parameters = parameters
    entity.depends = depends

    setattr(entity, "flang_source", str(event.get("source", "")))
    setattr(
        entity,
        "flang_body",
        {
            "locals": event.get("locals", []),
            "assignments": event.get("assignments", []),
            "calls": event.get("calls", []),
            "decls": event.get("decls", []),
            "stmts": event.get("stmts", []),
            "is_function": str(event.get("kind", "")) == "function_subprogram",
            "result_name": event.get("result_name"),
            "result_type": event.get("result_type"),
        },
    )
    return entity


def _merge_flang_entities(program: Program, flang_entities: List[Function]) -> None:
    """
    Merge Flang-derived entities into the program's existing entities.
    """
    by_name = {entity.name.lower(): entity for entity in flang_entities}

    for entity in program.entities:
        if not isinstance(entity, Function):
            continue
        sp = by_name.get(entity.name.lower())
        if sp is None:
            continue

        setattr(entity, "flang_source", getattr(sp, "flang_source", ""))
        setattr(entity, "flang_body", getattr(sp, "flang_body", None))
        entity.depends = set(getattr(entity, "depends", set()) or set()) | set(sp.depends)

    existing = {entity.name.lower() for entity in program.entities}
    for entity in flang_entities:
        if entity.name.lower() not in existing:
            program.entities.append(entity)


def _parse_arg_dat(loop: OP.Loop, opt: bool, args: List[Dict[str, Any]], loc: Location) -> None:
    expected = 7 if opt else 6
    if len(args) != expected:
        raise ParseError(
            f"incorrect number of arguments for op_{'opt_' if opt else ''}arg_dat", loc
        )

    args_list = args[1:] if opt else args

    dat_ptr = _arg_as_name(args_list[0], loc)
    map_idx = _arg_as_int(args_list[1], loc, optional=True)

    map_ptr: Optional[str] = _arg_as_name(args_list[2], loc)
    if map_ptr.upper() == "OP_ID":
        map_ptr = None

    dat_dim = _arg_as_int(args_list[3], loc, optional=True)

    dat_typ, dat_soa = parseType(_arg_as_string(args_list[4], loc), loc)
    access_type = _arg_as_access_type(args_list[5], loc)

    loop.addArgDat(loc, dat_ptr, dat_dim, dat_typ, dat_soa, map_ptr, map_idx, access_type, opt)


def _parse_arg_gbl(loop: OP.Loop, opt: bool, args: List[Dict[str, Any]], loc: Location) -> None:
    expected = 5 if opt else 4
    if len(args) != expected:
        raise ParseError(
            f"incorrect number of arguments for op_{'opt_' if opt else ''}arg_gbl", loc
        )

    args_list = args[1:] if opt else args

    ptr = _arg_as_name(args_list[0], loc)
    dim = _arg_as_int(args_list[1], loc, optional=True)
    typ = parseType(_arg_as_string(args_list[2], loc), loc)[0]
    access_type = _arg_as_access_type(args_list[3], loc)

    loop.addArgGbl(loc, ptr, dim, typ, access_type, opt)


def _parse_arg_idx(loop: OP.Loop, args: List[Dict[str, Any]], loc: Location) -> None:
    if len(args) != 2:
        raise ParseError("incorrect number of arguments for op_arg_idx", loc)

    map_idx = _arg_as_int(args[0], loc, optional=True)
    map_ptr: Optional[str] = _arg_as_name(args[1], loc)

    if map_ptr.upper() == "OP_ID":
        map_ptr = None

    loop.addArgIdx(loc, map_ptr, map_idx)


def _parse_arg_info(loop: OP.Loop, args: List[Dict[str, Any]], loc: Location) -> None:
    if len(args) != 4:
        raise ParseError("incorrect number of arguments for op_arg_info", loc)

    ptr = _arg_as_name(args[0], loc)
    dim = _arg_as_int(args[1], loc, optional=True)
    typ = parseType(_arg_as_string(args[2], loc), loc)[0]

    ref = _arg_as_int(args[3], loc)
    assert ref is not None

    loop.addArgInfo(loc, ptr, dim, typ, ref)


def _parse_const(args: List[Dict[str, Any]], loc: Location) -> OP.Const:
    if len(args) != 3:
        raise ParseError("incorrect number of arguments for op_decl_const", loc)

    ptr = _arg_as_name(args[0], loc)
    dim = _arg_as_int(args[1], loc)
    assert dim is not None

    typ_str = _arg_as_string(args[2], loc).strip().lower()
    typ = parseType(typ_str, loc)[0]

    return OP.Const(loc, ptr, dim, typ)


def _parse_loop(program: Program, args: List[Dict[str, Any]], loc: Location) -> OP.Loop:
    if len(args) < 3:
        raise ParseError("incorrect number of arguments for op_par_loop", loc)

    kernel = _arg_as_name(args[0], loc)
    name = f"{program.path.stem}_{len(program.loops) + 1}_{kernel}"
    loop = OP.Loop(name, loc, kernel)

    for arg_node in args[2:]:
        kind = arg_node.get("kind")

        if kind == "call":
            call_name = str(arg_node.get("name", "")).lower()
            sub_args = arg_node.get("args", [])

            if call_name == "op_arg_dat":
                _parse_arg_dat(loop, False, sub_args, loc)
            elif call_name == "op_opt_arg_dat":
                _parse_arg_dat(loop, True, sub_args, loc)
            elif call_name == "op_arg_gbl":
                _parse_arg_gbl(loop, False, sub_args, loc)
            elif call_name == "op_opt_arg_gbl":
                _parse_arg_gbl(loop, True, sub_args, loc)
            elif call_name == "op_arg_idx":
                _parse_arg_idx(loop, sub_args, loc)
            elif call_name == "op_arg_info":
                _parse_arg_info(loop, sub_args, loc)
            else:
                raise ParseError(f"invalid loop argument call: {call_name}", loc)

        else:
            raise ParseError(
                f"unable to parse op_par_loop argument (kind={kind})", loc
            )

    return loop


def build_program_from_flang(path: Path, source: str, data: Dict[str, Any]) -> Program:
    """
    Build a complete `Program` from the JSON emitted by op2-flang-scan.

    Populates loops, consts, and Function entities (each with `flang_source`,
    parameters, and raw dependency names). Cross-file resolution of known
    callees (and retention of unknown `CALL` names) is deferred to
    `resolve_flang_dependencies` once every translation unit has been
    loaded into the `Application`.
    """
    program = Program(path, None, source)
    setattr(program, "used_parser", "flang")

    for event in data.get("events", []):
        kind = str(event.get("kind", ""))
        loc = _make_loc(path, event.get("location"))

        if kind in _SUBPROGRAM_KINDS:
            name = str(event.get("name", "")).lower()
            if name:
                program.entities.append(_function_from_subprogram_event(event, program))
            continue

        args = event.get("args", [])

        if kind == "op_decl_const":
            program.consts.append(_parse_const(args, loc))
        elif _PAR_LOOP_RE.match(kind):
            program.loops.append(_parse_loop(program, args, loc))
        else:
            raise ParseError(f"unexpected event kind from flang-scan: {kind}", loc)

    return program


def parsed_with_flang(app: Application) -> bool:
    """
    Return True if any program in the application was parsed with Flang.
    """
    return any(getattr(p, "used_parser", "fparser2") == "flang" for p in app.programs)


def populate_program(program: Program, data: Dict[str, Any]) -> None:
    """
    Overlay Flang-derived data onto an existing fparser2-built program.
    """
    built = build_program_from_flang(program.path, program.source, data)
    program.loops = built.loops
    program.consts = built.consts
    _merge_flang_entities(program, built.entities)


def resolve_flang_dependencies(app: Application) -> None:
    """
    Resolve each entity's `depends` set against the application.

    The scanner records a superset of `CallStmt` names (always real calls) and
    `FunctionReference` names (real calls or array indexing, which Flang cannot
    tell apart without a symbol table). `CallStmt` names stay even if they do
    not resolve, whereas `FunctionReference` names stay only if they resolve to
    a `Function`.
    """
    known_names: Set[str] = set()
    for program in app.programs:
        for entity in program.entities:
            if isinstance(entity, Function):
                known_names.add(entity.name.lower())

    for program in app.programs:
        for entity in program.entities:
            if not isinstance(entity, Function):
                continue
            call_names = _flang_call_names(entity)
            entity.depends = {
                dep.lower()
                for dep in entity.depends
                if dep.lower() != entity.name.lower()
                and (dep.lower() in known_names or dep.lower() in call_names)
            }
