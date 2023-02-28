import cProfile
import dataclasses
import json
import os
import pstats
import pdb

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime
from pathlib import Path
from fnmatch import fnmatch

import cpp
import fortran
from jinja import env
from language import Lang
from op import OpError, Type
from scheme import Scheme
from store import Application, ParseError
from target import Target
from util import getVersion, safeFind


def main(argv=None) -> None:
    # Build arg parser
    parser = ArgumentParser(prog="op2-translator")

    # Flags
    parser.add_argument("-V", "--version", help="Version", action="version", version=getVersion())
    parser.add_argument("-v", "--verbose", help="Verbose", action="store_true")
    parser.add_argument("-d", "--dump", help="JSON store dump", action="store_true")
    parser.add_argument("-o", "--out", help="Output directory", type=isDirPath)
    parser.add_argument("-c", "--config", help="Target configuration", type=json.loads, default="{}")
    parser.add_argument("-soa", "--force_soa", help="Force Structs of Arrays", action="store_true")

    parser.add_argument("--suffix", help="Add a suffix to generated program translations", default="")
    parser.add_argument("--extra-consts-list", help="Extra consts to rename in kernels", default=None)
    parser.add_argument("--user-consts-module", help="Use a custom Fortran consts module", default=None)
    parser.add_argument(
        "--regex-program-translator", help="Use the regex-based program translator (Fortran only)", action="store_true"
    )

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

    # Invoke arg parser
    args = parser.parse_args(argv)

    if os.environ.get("OP_AUTO_SOA") is not None:
        args.force_soa = True

    file_parents = [Path(file_path).parent for file_path in args.file_paths]

    if args.out is None:
        args.out = file_parents[0]

    script_parents = list(Path(__file__).resolve().parents)
    if len(script_parents) >= 3 and script_parents[2].stem == "OP2-Common":
        args.I = [[str(script_parents[2].joinpath("op2/include"))]] + args.I

    args.I = [[str(file_parent)] for file_parent in dict.fromkeys(file_parents).keys()] + args.I

    # Collect the set of file extensions
    extensions = {str(Path(file_path).suffix)[1:] for file_path in args.file_paths}

    # Validate the file extensions
    if not extensions:
        exit("Missing file extensions, unable to determine target language.")
    elif len(extensions) > 1:
        exit("Varying file extensions, unable to determine target language.")
    else:
        [extension] = extensions

    lang = Lang.find(extension)

    if lang is None:
        exit(f"Unknown file extension: {extension}")

    if args.extra_consts_list is not None and hasattr(lang, "extra_consts_list"):
        lang.extra_consts_list = args.extra_consts_list

        if args.verbose:
            print(f"Using extra consts list: {lang.extra_consts_list}")

    if args.user_consts_module is not None and hasattr(lang, "user_consts_module"):
        lang.user_consts_module = args.user_consts_module

        if args.verbose:
            print(f"Using consts module: {lang.user_consts_module}")

    if args.regex_program_translator and hasattr(lang, "use_regex_translator"):
        lang.use_regex_translator = True

        if args.verbose:
            print(f"Using regex program translator")

    Type.set_formatter(lang.formatType)

    if len(args.target) == 0:
        args.target = [[target_name] for target_name in target_names]

    try:
        app = parse(args, lang)
    except ParseError as e:
        exit(e)

    if args.force_soa:
        for program in app.programs:
            for loop in program.loops:
                loop.dats = [dataclasses.replace(dat, soa=True) for dat in loop.dats]

    if args.verbose:
        print()
        print(app)

    # Validation phase
    try:
        validate(args, lang, app)
    except OpError as e:
        exit(e)

    for [target] in args.target:
        target = Target.find(target)

        for key in target.config:
            if key in args.config:
                target.config[key] = args.config[key]

        scheme = Scheme.find((lang, target))
        if not scheme:
            if args.verbose:
                print(f"No scheme registered for {lang}/{target}\n")

            continue

        if args.verbose:
            print(f"Translation scheme: {scheme}")

        codegen(args, scheme, app, args.force_soa)

        if args.verbose:
            print()

    # Generate program translations
    for i, program in enumerate(app.programs, 1):
        include_dirs = set([Path(dir) for [dir] in args.I])
        defines = [define for [define] in args.D]

        source = lang.translateProgram(program, include_dirs, defines, args.force_soa)

        new_file = os.path.splitext(os.path.basename(program.path))[0]
        ext = os.path.splitext(os.path.basename(program.path))[1]
        new_path = Path(args.out, f"{new_file}{args.suffix}{ext}")

        write_file(new_path, source)

        if args.verbose:
            print(f"Translated program {i} of {len(args.file_paths)}: {new_path}")


def write_file(path: Path, text: str) -> None:
    if path.is_file():
        prev_text = path.read_text()

        if text == prev_text:
            return

    with path.open("w") as f:
        # f.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by op2-translator\n\n")
        f.write(text)


def parse(args: Namespace, lang: Lang) -> Application:
    app = Application()

    # Collect the include directories
    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    # Parse the input files
    for i, raw_path in enumerate(args.file_paths, 1):
        if args.verbose:
            print(f"Parsing file {i} of {len(args.file_paths)}: {raw_path}")

        # Parse the program
        program = lang.parseProgram(Path(raw_path), include_dirs, defines)
        app.programs.append(program)

    return app


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

        if args.verbose:
            print("Dumped store:", store_path, end="\n\n")


def codegen(args: Namespace, scheme: Scheme, app: Application, force_soa: bool) -> None:
    # Collect the paths of the generated files
    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    # Generate loop hosts
    for i, (loop, program) in enumerate(app.loops(), 1):
        # Generate loop host source
        res = scheme.genLoopHost(env, loop, program, app, i)

        if res is None:
            print(f"Error: unable to generate loop host {i}")
            continue

        source, extension, fallback = res

        # Form output file path
        Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
        path = Path(
            args.out,
            scheme.target.name,
            f"{loop.name}_kernel.{extension}",
        )

        # Write the generated source file
        write_file(path, source)

        if args.verbose and not fallback:
            print(f"Generated loop host {i} of {len(app.loops())}: {path}")

        if args.verbose and fallback:
            print(f"Generated loop host {i} of {len(app.loops())} (fallback): {path}")

    # Generate consts file
    if scheme.consts_template is not None and getattr(scheme.lang, "user_consts_module", None) is None:
        source, name = scheme.genConsts(env, app)

        Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
        path = Path(args.out, scheme.target.name, name)

        write_file(path, source)

        if args.verbose:
            print(f"Generated consts file: {path}")

    # Generate master kernel file
    if scheme.master_kernel_template is not None:
        user_types_name = f"user_types.{scheme.lang.include_ext}"
        user_types_candidates = [Path(dir, user_types_name) for dir in include_dirs]
        user_types_file = safeFind(user_types_candidates, lambda p: p.is_file())

        source, name = scheme.genMasterKernel(env, app, user_types_file)

        Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
        path = Path(args.out, scheme.target.name, name)

        write_file(path, source)

        if args.verbose:
            print(f"Generated master kernel file: {path}")


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
    if os.environ.get("OP2_TRANSLATOR_PROFILE") is not None:
        profiler = cProfile.Profile()
        profiler.enable()

    main()

    if os.environ.get("OP2_TRANSLATOR_PROFILE") is not None:
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
