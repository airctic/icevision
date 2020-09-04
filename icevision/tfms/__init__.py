from icevision.tfms.transform import *

import icevision.tfms.batch

# Soft dependencies
from icevision.soft_dependencies import SoftDependencies

if SoftDependencies.albumentations:
    # `import icevision.tfms.albumentations as A` fails in python 3.6
    # https://stackoverflow.com/a/24968941/6772672
    from icevision.tfms import albumentations as A
