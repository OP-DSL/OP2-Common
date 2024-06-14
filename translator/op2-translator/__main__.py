import cProfile
import dataclasses
import json
import os
import pdb
import pstats
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from multiprocessing import Pool
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

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
    parser.add_argument("-c", "--config", help="Target configuration", action="append", type=json.loads, default=[])
    parser.add_argument("-soa", "--force_soa", help="Force Structs of Arrays", action="store_true")

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

    lang.parseArgs(args)

    Type.set_formatter(lang.formatType)

    if len(args.target) == 0:
        args.target = [[target_name] for target_name in target_names]

    include_dirs = set([Path(dir) for [dir] in args.I])
    defines = [define for [define] in args.D]

    try:
        app = parse(args, lang)
    except ParseError as e:
        print(e)
        exit(1)

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

    if args.verbose:
        print()
        print(app)

    # Validation phase
    try:
        print()
        print("Validating...")
        validate(args, lang, app)
    except OpError as e:
        print(e)
        exit(1)

    for [target] in args.target:
        target = Target.find(target)
        scheme = Scheme.find((lang, target))

        if not scheme:
            print(f"No scheme registered for {lang}/{target}\n")
            continue

        print(f"Translation scheme: {scheme}")
        codegen(args, scheme, app, args.force_soa)
        print()

    # Generate program translations
    for i, program in enumerate(app.programs, 1):
        source = lang.translateProgram(program, include_dirs, defines, args.force_soa)

        new_file = os.path.splitext(os.path.basename(program.path))[0]
        ext = os.path.splitext(os.path.basename(program.path))[1]
        new_path = Path(args.out, f"{new_file}{args.suffix}{ext}")

        write_file(new_path, source)

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
    f_args = [(i, raw_path, lang, args) for i, raw_path in enumerate(args.file_paths, 1)]

    print(f"Parsing files:")
    for raw_path in args.file_paths:
        print(f"    {raw_path}")

    app = Application()

    if lang.ast_is_serializable:
        app.programs = Pool().starmap(parse_file, f_args)
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

    # Generate loop hosts
    for i, (loop, program) in enumerate(app.loops(), 1):
        force_generate = scheme.target == Target.find("seq")

        # Generate loop host source
        res = scheme.genLoopHost(env, loop, program, app, i, args.config, force_generate)

        if res is None:
            print(f"Error: unable to generate loop host {i}")
            continue

        files, fallback = res

        Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
        for index, (source, extension) in enumerate(files):
            name = f"{loop.name}_kernel"
            if index > 0:
                name += f"_aux{index}"

            path = Path(
                args.out,
                scheme.target.name,
                f"{name}.{extension}",
            )

            write_file(path, source)

        if not fallback:
            print(f"Generated loop host {i} of {len(app.loops())}: {loop.name}")

        if fallback:
            loop.fallback = True
            print(f"Generated loop host {i} of {len(app.loops())} (fallback): {loop.name}")

    # Generate consts file
    if scheme.consts_template is not None and getattr(scheme.lang, "user_consts_module", None) is None:
        source, name = scheme.genConsts(env, app)

        Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)
        path = Path(args.out, scheme.target.name, name)

        write_file(path, source)
        print(f"Generated consts file: {path}")

    # Generate master kernel file
    if len(scheme.master_kernel_templates) > 0:
        user_types_name = f"user_types.{scheme.lang.include_ext}"
        user_types_candidates = [Path(dir, user_types_name) for dir in include_dirs]
        user_types_file = safeFind(user_types_candidates, lambda p: p.is_file())

        files = scheme.genMasterKernel(env, app, user_types_file)

        for index, (source, extension) in enumerate(files):
            Path(args.out, scheme.target.name).mkdir(parents=True, exist_ok=True)

            name = f"op2_kernels"
            if index > 0:
                name += f"_aux{index}"

            path = Path(args.out, scheme.target.name, f"{name}.{extension}")

            write_file(path, source)
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
    if os.environ.get("OP2_TRANSLATOR_PROFILE"):
        profiler = cProfile.Profile()

        profiler.enable()
        main()
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    else:
        main()
