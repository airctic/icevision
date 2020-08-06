from mantisshrimp.tfms.transform import *

import mantisshrimp.tfms.batch

from mantisshrimp.utils.soft_dependencies import HAS_ALBUMENTATIONS

if HAS_ALBUMENTATIONS:
    # `import mantisshrimp.tfms.albumentations as A` fails in python 3.6
    # https://stackoverflow.com/a/24968941/6772672
    from mantisshrimp.tfms import albumentations as A
