from mantisshrimp.imports import Path,json,plt
from mantisshrimp import *

source = Path(__file__).absolute().parent.parent.parent/'samples'

def test_image_info():
    annots_dict = json.loads((source / 'annotations.json').read())
    ainfo = annots_dict['images'][0]
    info = ImageInfo(ainfo['id'], ainfo['file_name'], ainfo['height'], ainfo['width'], 0)
    assert info == ImageInfo(128372, filepath='000000128372.jpg', h=427, w=640, split=0)

def test_info_parser():
    parser = test_utils.sample_info_parser()
    infos = parser.parse(show_pbar=False)
    assert len(infos) == 6
    assert infos[0] == ImageInfo(0, filepath=source/'000000128372.jpg', h=427, w=640, split=0)

def test_category_parser():
    catparser = test_utils.sample_category_parser()
    catmap = catparser.parse(show_pbar=False)
    assert catmap.cats[0].name == 'background'
    assert len(catmap) == 81
    assert catmap.cats[2] == Category(2, 'bicycle')
    assert catmap.id2i[42] == 38

def test_coco_annotation_parser():
    annotparser = test_utils.sample_annotation_parser()
    annots = annotparser.parse(show_pbar=False)
    annot = annots[0]
    assert len(annots) == 5
    assert annot.imageid == 0
    assert annot.labels == [4]

def test_coco_parser():
    parser = test_utils.sample_data_parser()
    with np_local_seed(42): train_rs,valid_rs = parser.parse(show_pbar=False)
    r = train_rs[0]
    assert len(train_rs)+len(valid_rs) == 5
    assert (r.info.h, r.info.w) == (427, 640)
    assert r.info.imageid == 0
    assert r.annot[0].bbox.xywh, [0.0, 73.89, 416.44, 305.13]
    assert r.info.filepath == source/'images/000000128372.jpg'

def test_show_record(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    parser = test_utils.sample_data_parser()
    with np_local_seed(42): train_rs, valid_rs = parser.parse(show_pbar=False)
    r = train_rs[0]
    show_record(r)

