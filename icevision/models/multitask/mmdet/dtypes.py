from icevision.imports import *
from icevision.models.multitask.utils.dtypes import *

ClassificationGroupDataDict = Dict[str, Union[List[str], Tensor, TensorDict]]
DataDictClassification = Dict[str, ClassificationGroupDataDict]
DataDictDetection = Union[
    TensorDict, ArrayDict, Dict[str, Union[Tuple[int], ImgMetadataDict]]
]
