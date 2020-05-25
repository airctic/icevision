from mantisshrimp.utils import *

def test_open_img():
    fn = '/home/lgvaz/git/mantisshrimp_py/samples/images/000000000089.jpg'
    assert open_img(fn).shape == (480,640,3)
    assert open_img(fn, gray=True).shape == (480,640)
