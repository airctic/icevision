__all__ = ['CategoryParser']

from ..imports import *
from ..utils import *
from ..core import *

class CategoryParser:
    def __init__(self, data): self.data = data
    def __iter__(self): yield from self.data
    def id(self, o): raise NotImplementedError
    def name(self, o): raise NotImplementedError

    def parse(self, show_pbar=True):
        return CategoryMap([Category(self.id(o), self.name(o)) for o in pbar(self, show=show_pbar)])

