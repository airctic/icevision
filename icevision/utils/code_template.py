__all__ = ["CodeLine", "CodeTemplate"]

from icevision.imports import *


@dataclass
class CodeLine:
    text: str
    indent: int


class CodeTemplate:
    def __init__(self, indent_spaces=4):
        self.lines: List[CodeLine] = []
        self.indent_str = " " * indent_spaces

    def add_line(self, line: str, indent=0):
        self.lines.append(CodeLine(line, indent))

    def add_lines(self, lines: Sequence[str], indent=0):
        for line in lines:
            self.add_line(line, indent)

    def display(self):
        for line in self.lines:
            print(f"{self.indent_str*line.indent}{line.text}")
