from icevision.imports import *
from icevision import *

# soft import icedata
try:
    import icedata
except ModuleNotFoundError as e:
    if str(e) != f"No module named 'icedata'":
        raise e
