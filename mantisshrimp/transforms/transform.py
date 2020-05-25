__all__ = ['Transform']

class Transform:
    def __init__(self, tfms): self.tfms = tfms
    def __call__(self, item):
        tfmed = self.apply(**item.asdict())
        return item.replace(**tfmed)
