from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# A place in the user's sources; column 0 when unknown.
@dataclass(frozen=True)
class Location:
    file: str
    line: int
    column: int = 0

    def __str__(self) -> str:
        return f"{self.file}:{self.line}" + (f":{self.column}" if self.column > 0 else "")


# A mistake in the user's sources or options: the translation stops, printing
# "file:line:col: error: message" and an optional note.  Anything else the
# translator raises is a bug in it.
class TranslatorError(Exception):
    def __init__(self, message: str, loc: Optional[Location] = None, note: Optional[Tuple[Location, str]] = None):
        super().__init__(message)
        self.message = message
        self.loc = loc
        self.note = note

    def __str__(self) -> str:
        text = f"{self.loc}: error: {self.message}" if self.loc is not None else f"error: {self.message}"
        if self.note is not None:
            text += f"\n{self.note[0]}: note: {self.note[1]}"
        return text

    # Raised in a multiprocessing worker, it is pickled back to the parent.
    def __reduce__(self):
        return (self.__class__, (self.message, self.loc, self.note))


# The sources are not valid input: a syntax error, or an OP2 call that does not parse.
class ParseError(TranslatorError):
    pass


# The sources use OP2 in a way the translator or a target does not support.
class OpError(TranslatorError):
    pass


# Where each line of a preprocessed text came from, read from the line
# markers the preprocessor writes ('#line 12 "file"' or '# 12 "file" 1'),
# which are taken out of the text.
class LineMap:
    _marker = re.compile(r'^#(?:line)?\s+(\d+)(?:\s+"([^"]*)")?')

    def __init__(self, path: str, origins: Optional[List[Tuple[str, int]]] = None):
        self.path = path
        self.origins = origins or []

    def locate(self, line: int, column: int = 0) -> Location:
        if 1 <= line <= len(self.origins):
            file, origin = self.origins[line - 1]
            return Location(file, origin, column)
        return Location(self.path, line, column)

    # The text without its markers, and its map.  pad=True gives a marker
    # within one file the blank lines it stands for, as pcpp writes them
    # without markers; pad=False drops it, as fpp -P does.
    @staticmethod
    def strip(text: str, path: str, pad: bool) -> Tuple[str, LineMap]:
        lines: List[str] = []
        origins: List[Tuple[str, int]] = []
        file, next_line = path, 1
        for line in text.split("\n"):
            marker = LineMap._marker.match(line)
            if marker is None:
                lines.append(line)
                origins.append((file, next_line))
                next_line += 1
                continue

            target = int(marker.group(1))
            if pad and origins and (marker.group(2) is None or marker.group(2) == file):
                for _ in range(max(0, target - next_line)):
                    lines.append("")
                    origins.append((file, next_line))
                    next_line += 1

            file = marker.group(2) if marker.group(2) is not None else file
            next_line = target

        return "\n".join(lines), LineMap(path, origins)
