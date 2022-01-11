from typing import Dict, List, Tuple, Union
from torch import Tensor
import numpy as np
import torch

__all__ = [
    "ImgMetadataDict",
    "TensorList",
    "TensorTuple",
    "TensorDict",
    "ArrayList",
    "ArrayDict",
]

ImgMetadataDict = Dict[str, Union[Tuple[int], np.ndarray]]
TensorList = List[Tensor]
TensorDict = Dict[str, Tensor]
TensorTuple = Tuple[Tensor]
ArrayList = List[np.ndarray]
ArrayDict = Dict[str, np.ndarray]
