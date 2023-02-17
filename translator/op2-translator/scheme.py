from __future__ import annotations

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

    consts_template: Optional[Path]
    loop_host_template: Path
    master_kernel_template: Optional[Path]

    def __str__(self) -> str:
        return f"{self.lang.name}/{self.target.name}"

    def canGenLoopHost(self, loop: OP.Loop) -> bool:
        return True

    def genLoopHost(
        self,
        include_dirs: Set[Path],
        defines: List[str],
        env: Environment,
        loop: OP.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
        force_generate: bool
    ) -> Tuple[str, str, bool]:
        # Load the loop host template
        template = env.get_template(str(self.loop_host_template))
        extension = self.loop_host_template.suffixes[-2][1:]

        if loop.fallback and not force_generate:
            return (None, extension, False)

        if not self.canGenLoopHost(loop):
            return (None, extension, False)

        try:
            kernel_func = self.translateKernel(loop, program, app, kernel_idx)
        except:
            print(f"Error: kernel translation for kernel {kernel_idx} failed ({self})")
            return (None, extension, False)

        # Generate source from the template
        return (
            template.render(
                OP=OP, lh=loop, kernel_func=kernel_func, kernel_idx=kernel_idx, lang=self.lang, target=self.target
            ),
            extension,
            True
        )

    def genConsts(self, env: Environment, app: Application) -> Tuple[str, str]:
        if self.consts_template is None:
            exit(f"No consts kernel template registered for {self}")

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
