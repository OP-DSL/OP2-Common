from __future__ import annotations

import sys
import traceback
import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from jinja2 import Environment

import op as OP
from language import Lang
from store import Application, Program
from target import Target
from util import Findable


class Scheme(Findable["Scheme"]):
    lang: Lang
    target: Target

    fallback: Optional[Scheme]

    consts_template: Optional[Path]
    loop_host_templates: List[Path]
    master_kernel_templates: List[Path]

    def __str__(self) -> str:
        return f"{self.lang.name}/{self.target.name}"

    def canGenLoopHost(self, loop: OP.Loop) -> bool:
        return True

    def getBaseConfig(self, loop: OP.Loop) -> Dict[str, Any]:
        return self.target.defaultConfig()

    def getConfig(self, loop: OP.Loop, config_overrides: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        config = self.getBaseConfig(loop)

        for override in config_overrides:
            for loop_match, items in override.items():
                if re.match(loop_match, loop.name):
                    config |= items

        return config

    def genLoopHost(
        self,
        env: Environment,
        loop: OP.Loop,
        program: Program,
        app: Application,
        kernel_idx: int,
        config_overrides: List[Dict[str, Dict[str, Any]]],
        force_generate: bool = False,
    ) -> Optional[Tuple[List[Tuple[str, str]], bool]]:
        def get_template(path):
            return env.get_template(str(path)), path.suffixes[-2][1:]

        main_templates = map(get_template, self.loop_host_templates)
        main_args = {
            "OP": OP,
            "lh": loop,
            "kernel_idx": kernel_idx,
            "lang": self.lang,
            "config": self.getConfig(loop, config_overrides),
            "kernel_func": None,
        }

        try:
            if (not loop.fallback and self.canGenLoopHost(loop)) or force_generate:
                main_args["kernel_func"] = self.translateKernel(loop, program, app, main_args["config"], kernel_idx)
        except Exception as e:
            print()
            print(f"Error: kernel translation for kernel {kernel_idx} ({loop.name}) failed ({self}):")
            print(f"  fallback: {loop.fallback}, can generate: {self.canGenLoopHost(loop)}, force_generate: {force_generate}")
            print(f"  {e}\n")

            if not isinstance(e, OP.OpError):
                traceback.print_exc(file=sys.stdout)
                print()

        if main_args["kernel_func"] is None and self.fallback is None:
            return None

        if self.fallback is not None:
            fallback_wrapper_template = get_template(self.lang.fallback_wrapper_template)
            fallback_templates = map(get_template, self.fallback.loop_host_templates)

            fallback_args = dict(main_args)

            fallback_args["config"] = self.fallback.getConfig(loop, config_overrides)
            fallback_args["kernel_func"] = self.fallback.translateKernel(
                loop, program, app, fallback_args["config"], kernel_idx
            )

        # Only fallback
        if main_args["kernel_func"] is None:
            rendered = [(t.render(**fallback_args, variant=""), ext) for t, ext in fallback_templates]
            return (rendered, True)

        # Only main
        if self.fallback is None:
            rendered = [(t.render(**main_args, variant=""), ext) for t, ext in main_templates]
            return (rendered, False)

        # Hybrid
        rendered_main     = [(t.render(**main_args,     variant="_main"),     ext) for t, ext in main_templates]
        rendered_fallback = [(t.render(**fallback_args, variant="_fallback"), ext) for t, ext in fallback_templates]

        rendered_hybrid = f"{rendered_main[0][0]}\n\n{rendered_fallback[0][0]}\n\n"
        rendered_hybrid += fallback_wrapper_template[0].render(**fallback_args)

        return ([(rendered_hybrid, fallback_wrapper_template[1])] + rendered_main[1:] + rendered_fallback[1:], False)

    def genConsts(self, env: Environment, app: Application) -> Tuple[str, str]:
        if self.consts_template is None:
            exit(f"No consts template registered for {self}")

        # Load the loop host template
        template = env.get_template(str(self.consts_template))

        extension = self.consts_template.suffixes[-2][1:]
        name = f"op2_consts.{extension}"

        # Generate source from the template
        return template.render(OP=OP, app=app, lang=self.lang, target=self.target), name

    def genMasterKernel(self, env: Environment, app: Application, user_types_file: Optional[Path]) -> List[Tuple[str, str]]:
        user_types = None
        if user_types_file is not None:
            user_types = user_types_file.read_text()

        files = []
        for template_path in self.master_kernel_templates:
            template = env.get_template(str(template_path))

            source = template.render(OP=OP, app=app, lang=self.lang, target=self.target, user_types=user_types)
            extension = template_path.suffixes[-2][1:]

            files.append((source, extension))

        return files

    @abstractmethod
    def translateKernel(
        self,
        loop: OP.Loop,
        program: Program,
        app: Application,
        config: Dict[str, Any],
        kernel_idx: int,
    ) -> str:
        pass

    def matches(self, key: Tuple[Lang, Target]) -> bool:
        if not (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], Lang) and isinstance(key[1], Target)):
            return False

        return self.lang == key[0] and self.target == key[1]
