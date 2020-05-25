__all__ = ['DataParser']

from ..imports import *
from ..core import *

@funcs_kwargs
class DataParser:
    _methods = 'info_parser annotation_parser'.split()
    def __init__(self, data, source, catmap=None, **kwargs):
        self.data,self.source,self.catmap = data,source,catmap

    def category_parser(self, data):                          raise NotImplementedError
    def info_parser(self, data, source):                      raise NotImplementedError
    def annotation_parser(self, data, source, catmap, idmap): raise NotImplementedError

    def parse(self, show_pbar=True):
        catmap = self.catmap or self.category_parser(self.data).parse(show_pbar)
        info_parser = self.info_parser(self.data, self.source)
        annotation_parser = self.annotation_parser(self.data, self.source, catmap=catmap, idmap=info_parser.idmap)
        imgs = L(info_parser.parse(show_pbar))
        annots = L(annotation_parser.parse(show_pbar))
        # Remove imgs that don't have annotations
        img_iids = set(imgs.attrgot('imageid'))
        valid_iids = set(annots.attrgot('imageid'))
        if not valid_iids.issubset(img_iids):
            raise ValueError(f'imageids {valid_iids - img_iids} present in annotations but not in images')
        valid_imgs = imgs.filter(lambda o: o.imageid in valid_iids)
        print(f"Removed {len(imgs) - len(valid_iids)} images that don't have annotations")
        # Sort and get items
        assert len(annots) == len(valid_imgs)
        valid_imgs.sort(attrgetter('imageid'))
        annots.sort(attrgetter('imageid'))
        records = defaultdict(list)
        for iinfo, annot in zip(valid_imgs, annots):
            records[iinfo.split].append(Record(iinfo, annot))
        return list(records.values())
        # return [records[k] for k in records.keys()]
