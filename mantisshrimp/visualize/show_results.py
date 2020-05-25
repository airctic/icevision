__all__ = ['show_results']

from ..imports import *
from .show_preds import *

def show_results(learn, k=5):
    rs = random.choices(learn.valid_dl.dataset.records, k=k)
    show_preds(*learn.m.predict(rs=rs))
