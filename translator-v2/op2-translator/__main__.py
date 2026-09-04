import cProfile
import dataclasses
import json
import logging
import os
import pdb
import pstats
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from multiprocessing import Pool
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cpp
import fortran
from jinja import env
from language import Lang
from op import OpError, Type
from scheme import Scheme
from store import Application, ParseError
from target import Target
from util import getVersion, safeFind

logger = logging.getLogger(__name__)


def configure_logging(args: Namespace) -> None:
    console_level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    log_path = Path(args.out, "op2-translator.log")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # must be <= the lowest handler level (DEBUG) or the file handler starves
    root.addHandler(console_handler)
    root.addHandler(file_handler)


def main(argv=None) -> None:
    # Build arg parser
    parser = ArgumentParser(prog="op2-translator")

    # Flags
    parser.add_argument("-V", "--version", help="Version", action="version", version=getVersion())
    parser.add_argument("-v", "--verbose", help="Increase verbosity (-v info, -vv debug)", action="count", default=0)
    parser.add_argument("-d", "--dump", help="JSON store dump", action="store_true")
    parser.add_argument("-o", "--out", help="Output directory", type=isDirPath)
    parser.add_argument("--depfile", help="Write a Make-style depfile of the sources read", type=str)
    parser.add_argument("-c", "--config", help="Target configuration", action="append", type=json.loads, default=[])
    parser.add_argument("-soa", "--force_soa", help="Force Structs of Arrays", action="store_true")
    parser.add_argument("-mp", "--multiprocess_parse", help="Force Multiprocess Parsing", action="store_true")

    parser.add_argument("--suffix", help="Add a suffix to generated program translations", default="")

    parser.add_argument("-I", help="Add to include directories", type=isDirPath, action="append", nargs=1, default=[])
    parser.add_argument("-D", help="Add to preprocessor defines", action="append", nargs=1, default=[])

    target_names = [target.name for target in Target.all()]
    parser.add_argument(
        "-t",
        "--target",
        help="Code-generation target",
        type=str,
        action="append",
        nargs=1,
        choices=target_names,
        default=[],
    )

    parser.add_argument("file_paths", help="Input OP2 sources", type=isFilePath, nargs="+")

    for lang in Lang.all():
        lang.addArgs(parser)

    # Invoke arg parser
    args = parser.parse_args(argv)

    if os.environ.get("OP_AUTO_SOA") is not None:
        args.force_soa = True

    file_parents = [Path(file_path).parent for file_path in args.file_paths]

    if args.out is None:
        args.out = file_parents[0]

    configure_logging(args)

    script_parents = list(Path(__file__).resolve().parents)
    if len(script_parents) >= 3 and script_parents[2].stem == "OP2-Common":
        args.I = [[str(script_parents[2].joinpath("op2/include"))]] + args.I

    args.I = [[str(file_parent)] for file_parent in dict.fromkeys(file_parents).keys()] + args.I

    # Collect the set of file extensions
    extensions = {str(Path(file_path).suffix)[1:] for file_path in args.file_paths}

    # Validate the file extensions
    if not extensions:
        logger.error("Missing file extensions, unable to determine target language.")
        sys.exit(1)
    elif len(extensions) > 1:
        logger.error("Varying file extensions, unable to determine target language.")
        sys.exit(1)
    else:
        [extension] = extensions

    lang = Lang.find(extension)

    if lang is None:
        logger.error(f"Unknown file extension: {extension}")
        sys.exit(1)

    lang.parseArgs(args)

    Type.set_formatter(lang.formatType)

    if len(args.target) == 0:
        args.target = [[target_name] for target_name in target_names]

    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    try:
        app = parse(args, lang)
    except ParseError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.consts_module is not None:
        app.consts_module = lang.parseProgram(Path(args.consts_module), include_dirs, defines)

    if args.extra_consts_list is not None:
        with open(args.extra_consts_list, "r") as f:
            for line in f:
                const_ptr = line.strip()

                if const_ptr != "":
                    app.external_consts.add(const_ptr.lower())

    if args.force_soa:
        for program in app.programs:
            for loop in program.loops:
                loop.dats = [dataclasses.replace(dat, soa=True) for dat in loop.dats]

    if args.verbose >= 2:
        logger.debug("%s", app)

    # Validation phase
    try:
        logger.info("Validating...")
        validate(args, lang, app)
    except OpError as e:
        logger.error(str(e))
        sys.exit(1)

    for [target] in args.target:
        target = Target.find(target)
        scheme = Scheme.find((lang, target))

        if not scheme:
            logger.warning(f"No scheme registered for {lang}/{target}")
            continue

        logger.info(f"Translation scheme: {scheme}")
        codegen(args, scheme, app, args.force_soa)

    # Generate program translations
    for i, program in enumerate(app.programs, 1):
        source = lang.translateProgram(program, include_dirs, defines, args.force_soa)

        new_file = os.path.splitext(os.path.basename(program.path))[0]
        ext = os.path.splitext(os.path.basename(program.path))[1]
        new_path = Path(args.out, f"{new_file}{args.suffix}{ext}")

        write_file(new_path, source, args)

        logger.info(f"Translated program {i} of {len(args.file_paths)}: {new_path}")

    if args.depfile is not None:
        write_depfile(Path(args.depfile), generated_paths, app)


def escape_depfile_path(path: Path) -> str:
    # Per the depfile grammar: '$' doubles, '#' and ' ' are backslash-escaped.
    return str(path).replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ")


def write_depfile(path: Path, targets: List[Path], app: Application) -> None:
    # One rule naming every generated file, so whichever of them the build
    # system asks about is present. Absolute paths throughout, which sidesteps
    # the question of what a relative path in a depfile is relative to.
    #
    # With no targets there is no rule to write, but the file is still
    # truncated rather than left alone: an earlier run's rule left in place
    # would go on feeding the build system dependencies this run never
    # claimed. Both Make and Ninja accept an empty depfile: it states no
    # rule, so neither can conclude anything is up to date from it, and the
    # fallback is a rebuild rather than a wrong answer.
    if len(targets) == 0:
        path.write_text("")
        logger.info(f"Wrote empty depfile: {path} (nothing generated)")
        return

    dependencies: Set[Path] = set()
    for program in list(app.programs) + ([app.consts_module] if app.consts_module is not None else []):
        dependencies.add(Path(program.path).resolve())
        dependencies |= program.includes

    rule = " ".join(escape_depfile_path(target.resolve()) for target in targets) + ":"
    for dependency in sorted(dependencies):
        rule += " \\\n    " + escape_depfile_path(dependency)

    path.write_text(rule + "\n")
    logger.info(f"Wrote depfile: {path} ({len(dependencies)} dependencies)")


# Every path write_file() is asked for, whether or not the content turned out
# to be identical - these are the depfile's targets.
generated_paths: List[Path] = []


def write_file(path: Path, text: str, args: Namespace) -> None:
    if path.exists():
        for input_path in args.file_paths:
            if not path.samefile(input_path):
                continue

            logger.error(f"generating file '{path}' would overwrite input file\nPass an output directory with -o <path>")
            sys.exit(1)

    generated_paths.append(path)

    if path.is_file():
        prev_text = path.read_text()

        if text == prev_text:
            return

    # Write-then-rename, not a plain open("w"): truncate-then-write leaves a
    # window where a reader sees a half-written file, and the build system may
    # run translator processes over one output directory in parallel.
    # os.replace is atomic within a filesystem, and the pid in the temporary
    # name keeps two writers off the same scratch file.
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def parse(args: Namespace, lang: Lang) -> Application:
    f_args = [(i, raw_path, lang, args) for i, raw_path in enumerate(args.file_paths, 1)]

    logger.info("Parsing files:\n" + "\n".join(f"    {p}" for p in args.file_paths))

    app = Application()

    if lang.ast_is_serializable:
        try:
            # Logging handlers are configured before this Pool() is constructed, so forked
            # workers inherit them (fork start method, the Linux default). parse_file() has
            # no logging calls today; if that changes, be aware forked workers would hold
            # duplicated file descriptors over the same log file with no cross-process
            # write synchronization, risking interleaved lines under -mp.
            if args.multiprocess_parse:
                app.programs = Pool().starmap(parse_file, f_args)
            else:
                app.programs = [parse_file(*args) for args in f_args]
        except fortran.FortranSyntaxError as err:
            logger.error(f"Syntax error in file {err.filename}:\n{err.message}")
            sys.exit(1)
    else:
        app.programs = []
        for a in f_args:
            app.programs.append(parse_file(*a))

    return app


def parse_file(i, raw_path, lang, args):
    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    return lang.parseProgram(Path(raw_path), include_dirs, defines)


def validate(args: Namespace, lang: Lang, app: Application) -> None:
    # Run semantic checks on the application
    app.validate(lang)

    # Create a JSON dump
    if args.dump:
        store_path = Path(args.out, "store.json")
        serializer = lambda o: getattr(o, "__dict__", "unserializable")

        # Write application dump
        with open(store_path, "w") as file:
            file.write(json.dumps(app, default=serializer, indent=4))

        print("Dumped store:", store_path, end="\n\n")


def codegen(args: Namespace, scheme: Scheme, app: Application, force_soa: bool) -> None:
    # Collect the paths of the generated files
    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    fallback_loops = {}

    # Generate loop hosts.
    #
    # Per-loop outputs are written as N separate files (one per op_par_loop
    # call), preserving source-level navigation, per-loop debugging line
    # numbers, and independent editability.  How the master compile unit
    # brings them in depends on the language:
    #
    #   * C++ variants: master `#include`s each per-loop `.hpp` / `.h` /
    #     `.cuh` / `.hip.h` file - headers, resolved at compile time.
    #   * Fortran variants: master `#include`s each per-loop `.F90` file
    #     via CPP preprocessing (Fortran files with capital .F90 are
    #     preprocessed by every mainstream Fortran compiler).  Per-loop
    #     files define Fortran modules; concatenation via `#include` puts
    #     all modules in the master's compilation unit.
    #
    # Fortran per-loop templates declare various extensions (.F90, .CUF,
    # .inc) in the scheme, but the fallback-wrapper mechanic can substitute
    # extensions on hybrid loops.  We normalize every Fortran per-loop
    # output to `.F90` so the master template's `#include` line is
    # unambiguous - `.CUF` semantics (CUDA Fortran) are covered by the
    # `-cuda` compile flag we pass to nvfortran on the app target.
    #
    # C++ helper compile units (currently only c_seq's per-loop `.cpp`) are
    # still amalgamated - c_seq isn't wired up in the CMake build so this
    # code path is exercised only by direct translator use.
    FORTRAN_COMPILE_EXTENSIONS = {".F90", ".CUF", ".inc"}
    PER_LOOP_HEADER_EXTENSIONS = {".hpp", ".h", ".cuh", ".hip.h", ".mod"}

    per_loop_buffers: Dict[Tuple[int, str], List[str]] = {}

    for i, (loop, program) in enumerate(app.loops(), 1):
        force_generate = scheme.target == Target.find("seq")

        # Generate loop host source
        res = scheme.genLoopHost(env, loop, program, app, i, args.config, force_generate)

        if res is None:
            logger.warning(f"unable to generate loop host {i}")
            continue

        files, fallback = res

        Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
        for index, (source, extension) in enumerate(files):
            if extension in FORTRAN_COMPILE_EXTENSIONS:
                extension = ".F90"  # normalize Fortran per-loop to a uniform extension
                per_loop = True
            elif extension in PER_LOOP_HEADER_EXTENSIONS:
                per_loop = True
            else:
                per_loop = False

            if per_loop:
                name = f"{loop.name}_kernel"
                if index > 0:
                    name += f"_aux{index}"
                path = Path(args.out, scheme.target.name, f"{name}{extension}")
                write_file(path, source, args)
            else:
                per_loop_buffers.setdefault((index, extension), []).append(source)

        if not fallback:
            fallback_loops[loop.name] = False
            logger.info(f"Generated loop host {i} of {len(app.loops())}: {loop.name}")

        if fallback:
            fallback_loops[loop.name] = True
            logger.warning(f"Generated loop host {i} of {len(app.loops())} (fallback): {loop.name}")

    # Write the amalgamated per-loop compile-unit files (only c_seq's C++
    # helpers reach this path today).
    for (index, extension), sources in per_loop_buffers.items():
        name = "op2_loop_kernel"
        if index > 0:
            name += f"_aux{index}"
        path = Path(args.out, scheme.target.name, f"{name}{extension}")
        write_file(path, "\n".join(sources), args)

    # Generate consts file
    if scheme.consts_template is not None and getattr(scheme.lang, "user_consts_module", None) is None:
        source, name = scheme.genConsts(env, app)

        Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
        path = Path(args.out, scheme.target.name, name)

        write_file(path, source, args)
        logger.info(f"Generated consts file: {path}")

    # Generate master kernel file
    if len(scheme.master_kernel_templates) > 0:
        user_types_name = f"user_types.{scheme.lang.include_ext}"
        user_types_candidates = [Path(dir, user_types_name) for dir in include_dirs]
        user_types_file = safeFind(user_types_candidates, lambda p: p.is_file())

        files = scheme.genMasterKernel(env, app, user_types_file, fallback_loops)

        for index, (source, extension) in enumerate(files):
            Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)

            name = f"op2_kernels"
            if index > 0:
                name += f"_aux{index}"

            path = Path(args.out, scheme.target.name, f"{name}{extension}")

            write_file(path, source, args)
            logger.info(f"Generated master kernel file: {path}")


def isDirPath(path):
    if os.path.isdir(path):
        return path
    else:
        raise ArgumentTypeError(f"invalid dir path: {path}")


def isFilePath(path):
    if os.path.isfile(path):
        return path
    else:
        raise ArgumentTypeError(f"invalid file path: {path}")


if __name__ == "__main__":
    if os.environ.get("OP2_TRANSLATOR_PROFILE"):
        profiler = cProfile.Profile()

        profiler.enable()
        main()
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    else:
        main()
