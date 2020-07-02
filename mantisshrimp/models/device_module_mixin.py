__all__ = ["DeviceModuleMixin"]

from mantisshrimp.imports import *


class DeviceModuleMixin(nn.Module):
    @property
    def device(self):
        """ Returns the device the first model parameter is stored.

        Can be wrong if different parts of the model are in different devices.
        """
        return next(iter(self.parameters())).device
