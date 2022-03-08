import dataclasses
import json
import os
import re
import traceback
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime
from pathlib import Path
from typing import List

from jinja import env
from language import Lang
from op import OpError, Type
from optimisation import Opt
from scheme import Scheme
from store import Application, ParseError
from util import getVersion, safeFind


def main(argv=None) -> None:
    # Build arg parser
    parser = ArgumentParser(prog="op2-translator")

    # Flags
    parser.add_argument("-V", "--version", help="Version", action="version", version=getVersion())
    parser.add_argument("-v", "--verbose", help="Verbose", action="store_true")
    parser.add_argument("-d", "--dump", help="JSON store dump", action="store_true")
    parser.add_argument("-o", "--out", help="Output directory", type=isDirPath)
    parser.add_argument("-c", "--config", help="Optimisation configuration", type=json.loads, default="{}")
    parser.add_argument("-soa", "--force_soa", help="Force Structs of Arrays", action="store_true")

    parser.add_argument("-I", help="Header Include", type=isDirPath, action="append", nargs=1, default=[])

    opt_names = [opt.name for opt in Opt.all()]
    parser.add_argument(
        "-f", "--flavour", help="Optimisation scheme", type=str, action="append", nargs=1, choices=opt_names, default=[]
    )

    parser.add_argument("file_paths", help="Input OP2 sources", type=isFilePath, nargs="+")

    # Invoke arg parser
    args = parser.parse_args(argv)

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

    # Determine the target language and optimisation
    lang = Lang.find(extension)

    if lang is None:
        exit(f"Unknown file extension: {extension}")

    Type.set_formatter(lang.formatType)

    if len(args.flavour) == 0:
        args.flavour = [[opt_name] for opt_name in opt_names]

    for [flavour] in args.flavour:
        opt = Opt.find(flavour)

        for key in opt.config:
            if key in args.config:
                opt.config[key] = args.config[key]

        scheme = Scheme.find((lang, opt))
        if not scheme:
            print(f"No scheme registered for {lang}/{opt}")
            continue

        if args.verbose:
            print(f"Translation scheme: {scheme}")

        run(args, scheme)
        if args.verbose:
            print()


def run(args: Namespace, scheme: Scheme) -> None:
    # Parsing phase
    try:
        app = parse(args, scheme)
    except ParseError as e:
        exit(e)

    if args.force_soa:
        for program in app.programs:
            program.dats = [dataclasses.replace(dat, soa=True) for dat in program.dats]

    if args.verbose:
        print()
        print(app)

    # Validation phase
    try:
        validate(args, scheme, app)
    except OpError as e:
        exit(e)

    # Code-generation phase
    codegen(args, scheme, app, args.force_soa)


def parse(args: Namespace, scheme: Scheme) -> Application:
    app = Application()

    # Collect the include directories
    include_dirs = set([Path(dir) for [dir] in args.I])

    # Parse the input files
    for i, raw_path in enumerate(args.file_paths, 1):
        if args.verbose:
            print(f"Parsing file {i} of {len(args.file_paths)}: {raw_path}")

        # Parse the program
        program = scheme.lang.parseProgram(Path(raw_path), include_dirs)
        app.programs.append(program)

    # Parse the referenced kernels
    for kernel_name in {loop.kernel for loop in app.loops()}:
        kernel_include_name = f"{kernel_name}.{scheme.lang.include_ext}"
        kernel_include_files = [Path(dir, kernel_include_name) for dir in include_dirs]
        kernel_include_files = list(filter(lambda p: p.is_file(), kernel_include_files))

        for path in [Path(raw_path) for raw_path in args.file_paths] + kernel_include_files:
            kernel = scheme.lang.parseKernel(path, kernel_name, include_dirs)

            if kernel is not None:
                app.kernels[kernel_name] = kernel
                break

        if kernel_name not in app.kernels:
            exit(f"Failed to locate kernel function: {kernel_name}")

    return app


def validate(args: Namespace, scheme: Scheme, app: Application) -> None:
    # Run semantic checks on the application
    app.validate(scheme.lang)

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

    # Generate loop hosts
    for i, loop in enumerate(app.loops(), 1):
        # Generate loop host source
        source, extension = scheme.genLoopHost(include_dirs, env, loop, app, i)

        # Form output file path
        path = None
        if scheme.lang.kernel_dir:
            Path(args.out, scheme.opt.name).mkdir(parents=True, exist_ok=True)
            path = Path(
                args.out,
                scheme.opt.name,
                f"{loop.kernel}_kernel.{extension}",
            )
        else:
            path = Path(args.out, f"{loop.kernel}_{scheme.opt.name}_kernel.{extension}")

        # Write the generated source file
        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by op2-translator\n\n")
            file.write(source)

            if args.verbose:
                print(f"Generated loop host {i} of {len(app.loops())}: {path}")

    # Generate master kernel file
    if scheme.master_kernel_template != None:
        user_types_name = f"user_types.{scheme.lang.include_ext}"
        user_types_candidates = [Path(dir, user_types_name) for dir in include_dirs]
        user_types_file = safeFind(user_types_candidates, lambda p: p.is_file())

        source, extension = scheme.genMasterKernel(env, app, user_types_file)
        appname = os.path.splitext(os.path.basename(app.programs[0].path))[0]

        path = None
        if scheme.lang.kernel_dir:
            Path(args.out, scheme.opt.name).mkdir(parents=True, exist_ok=True)
            path = Path(args.out, scheme.opt.name, f"{appname}_kernels.{extension}")
        else:
            path = Path(args.out, f"{appname}_{scheme.opt.name}_kernels.{extension}")

        with open(path, "w") as file:
            file.write(f"{scheme.lang.com_delim} Auto-generated at {datetime.now()} by op2-translator\n\n")
            file.write(source)

            if args.verbose:
                print(f"Generated master kernel file: {path}")

    # Generate program translations
    for i, program in enumerate(app.programs, 1):
        # Read the raw source file
        with open(program.path, "r") as raw_file:

            # Generate the source translation
            source = scheme.lang.translateProgram(raw_file.read(), program, force_soa)

            # Form output file path
            new_file = os.path.splitext(os.path.basename(program.path))[0]
            ext = os.path.splitext(os.path.basename(program.path))[1]
            new_path = Path(args.out, f"{new_file}_op{ext}")

            # Write the translated source file
            with open(new_path, "w") as new_file:
                new_file.write(f"\n{scheme.lang.com_delim} Auto-generated at {datetime.now()} by op2-translator\n\n")
                new_file.write(source)

                if args.verbose:
                    print(f"Translated program  {i} of {len(args.file_paths)}: {new_path}")


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
    main()