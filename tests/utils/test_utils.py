from icevision.all import *


def test_notnone():
    assert not notnone(None)
    assert notnone(1)
    assert notnone("")
    assert notnone([])


def test_ifnotnone():
    assert ifnotnone(1, lambda o: o + 1) == 2
    assert ifnotnone(None, lambda o: o + 1) == None


def test_last():
    l = [1, 2, 1, 0]
    assert last(l) == l[-1]


def test_cleandict():
    d = {"a": 1, "b": 0, "c": None}
    assert cleandict(d) == {"a": 1, "b": 0}


def test_allequal():
    assert allequal([3, 3, 3]) == True
    assert allequal([]) == True
    assert allequal([1, 2, 3]) == False


def test_mergeds():
    ds = [{"a": 2}, {"b": 3}, {"a": 1}, {"c": 0}, {"b": 5}, {"a": 3}]
    assert mergeds(ds) == {"a": [2, 1, 3], "b": [3, 5], "c": [0]}


def normalize_denormalize():
    img = np.linspace(0, 255, 4 * 2 * 3, dtype=np.uint8).reshape(4, 2, 3)
    mean = img.mean() / 255
    std = img.std() / 255

    res = denormalize(normalize(img, mean=mean, std=std), mean=mean, std=std)
    assert type(res) == type(img)
    assert np.all(res == img)
