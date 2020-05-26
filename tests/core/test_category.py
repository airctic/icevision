from mantisshrimp.core import *


def test_category():
    cat = Category(0, "cat1")
    assert cat.id == 0
    assert cat.name == "cat1"


def test_category_equal():
    cat1, cat2, cat3 = Category(0, "cat1"), Category(1, "cat2"), Category(0, "cat1")
    assert not cat1 == cat2
    assert cat1 == cat3


def test_category_map():
    cat1, cat2, cat3 = Category(0, "cat1"), Category(1, "cat2"), Category(0, "cat1")
    catbkg = Category(name="background")
    catmap = CategoryMap([cat1, cat2, cat3])
    assert catmap.i2o == {0: catbkg, 1: cat1, 2: cat2, 3: cat3}
    assert catmap.cats == [catbkg, cat1, cat2, cat3]
    assert catmap.id2i == {0: 3, 1: 2}
    assert catmap.id2o == {0: cat3, 1: cat2}
