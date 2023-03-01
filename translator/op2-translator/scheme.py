from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional, Set, Tuple

from jinja2 import Environment

import op as OP
from language import Lang
from store import Application, Program
from target import Target
from util import Findable


class Scheme(Findable):
    lang: Lang
    target: Target

    fallback: Optional[Scheme]

    consts_template: Optional[Path]
    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return f"{self.lang.name}/{self.target.name}"

    def canGenLoopHost(self, loop: OP.Loop) -> bool:
        return True

    def genLoopHost(
        self,
        env: Environment,
        loop: OP.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> Optional[Tuple[str, str, bool]]:
        template = env.get_template(str(self.loop_host_template))
        extension = self.loop_host_template.suffixes[-2][1:]

        args = {
            "OP": OP,
            "lh": loop,
            "kernel_idx": kernel_idx,
            "lang": self.lang,
            "target": self.target,
        }

        cant_generate = not self.canGenLoopHost(loop)

        if (loop.fallback or cant_generate) and self.fallback is None:
            return None

        if self.fallback is not None:
            fallback_wrapper_template = env.get_template(str(self.lang.fallback_wrapper_template))

            fallback_template = env.get_template(str(self.fallback.loop_host_template))
            fallback_kernel_func = self.fallback.translateKernel(loop, program, app, kernel_idx)

        try:
            kernel_func = self.translateKernel(loop, program, app, kernel_idx)
        except Exception as e:
            print(f"Error: kernel translation for kernel {kernel_idx} failed ({self}):")
            traceback.print_exc()

            if self.fallback is None:
                return None

            kernel_func = None

        if loop.fallback or cant_generate or kernel_func is None:
            return (fallback_template.render(**args, variant="", kernel_func=fallback_kernel_func), extension, True)

        if self.fallback is None:
            return (template.render(**args, variant="", kernel_func=kernel_func), extension, False)

        source = template.render(**args, variant="_main", kernel_func=kernel_func)

        source += "\n\n"
        source += fallback_template.render(**args, variant="_fallback", kernel_func=fallback_kernel_func)

        source += "\n\n"
        source += fallback_wrapper_template.render(**args)

        return (source, extension, False)

    def genConsts(self, env: Environment, app: Application) -> Tuple[str, str]:
        if self.consts_template is None:
            exit(f"No consts template registered for {self}")

        # Load the loop host template
        template = env.get_template(str(self.consts_template))

        extension = self.consts_template.suffixes[-2][1:]
        name = f"op2_consts.{extension}"

        # Generate source from the template
        return template.render(OP=OP, app=app, lang=self.lang, target=self.target), name

    def genMasterKernel(self, env: Environment, app: Application, user_types_file: Optional[Path]) -> Tuple[str, str]:
        if self.master_kernel_template is None:
            exit(f"No master kernel template registered for {self}")

        user_types = None
        if user_types_file is not None:
            user_types = user_types_file.read_text()

        # Load the loop host template
        template = env.get_template(str(self.master_kernel_template))

        extension = self.master_kernel_template.suffixes[-2][1:]
        name = f"op2_kernels.{extension}"

        # Generate source from the template
        return template.render(OP=OP, app=app, lang=self.lang, target=self.target, user_types=user_types), name

    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
    ) -> str:
        pass

    def matches(self, key: Tuple[Lang, Target]) -> bool:
        return self.lang == key[0] and self.target == key[1]
