from typing import Any, Dict, Tuple

from store import Application, Kernel


def translateKernel(config: Dict[str, Any], source: str, kernel: Kernel, app: Application) -> str:
    return source
