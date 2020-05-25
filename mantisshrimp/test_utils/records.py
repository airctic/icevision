__all__ = ['sample_category_parser', 'sample_info_parser', 'sample_annotation_parser', 'sample_data_parser',
           'sample_records', 'sample_datasets', 'sample_rcnn_dataloaders']

from mantisshrimp.all import *

source = Path(__file__).absolute().parent.parent.parent/'samples'
annots_dict = json.loads((source/'annotations.json').read())

def sample_category_parser():
    return COCOCategoryParser(annots_dict['categories'])

def sample_info_parser():
    return COCOInfoParser(annots_dict['images'], source=source)

def sample_annotation_parser():
    catmap = sample_category_parser().parse(show_pbar=False)
    return COCOAnnotationParser(annots_dict['annotations'], source/'images', catmap)

def sample_data_parser():
    return COCOParser(annots_dict, source/'images')

def sample_records():
    with np_local_seed(42):
        return sample_data_parser().parse(show_pbar=False)

def sample_datasets():
    train_rs, valid_rs = sample_records()
    return Dataset(train_rs), Dataset(valid_rs)

def sample_rcnn_dataloaders():
    train_ds,valid_ds = sample_datasets()
    train_dl = RCNNDataLoader(train_ds, batch_size=2, shuffle=False, drop_last=False)
    valid_dl = RCNNDataLoader(valid_ds, batch_size=4, shuffle=False, drop_last=False)
    return train_dl, valid_dl
