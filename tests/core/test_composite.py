import pytest
from icevision.all import *


class MockComponent1(Component):
    order = 0.9

    def foo(self):
        return ["foo1"]

    def bar(self):
        return {"bar1": 1}


class MockComponent2(Component):
    order = 0.3

    def foo(self):
        return ["foo2"]

    def bar(self):
        return {"bar2": 2}


# TODO: test returned values
@pytest.mark.parametrize(
    "components_list,fn_name,reduction,expected",
    (
        ([MockComponent1(), MockComponent2()], "foo", "extend", ["foo2", "foo1"]),
        ([MockComponent1()], "foo", "extend", ["foo1"]),
        ([MockComponent1(), MockComponent2()], "foo", None, [["foo2"], ["foo1"]]),
        ([MockComponent1()], "foo", None, [["foo1"]]),
        ([MockComponent1(), MockComponent2()], "bar", "update", {"bar1": 1, "bar2": 2}),
        ([MockComponent1()], "bar", "update", {"bar1": 1}),
        ([MockComponent1(), MockComponent2()], "bar", None, [{"bar2": 2}, {"bar1": 1}]),
        ([MockComponent1()], "bar", None, [{"bar1": 1}]),
    ),
)
def test_reduce_on_components(components_list, fn_name, reduction, expected):
    composite = Composite(components_list)
    res = composite.reduce_on_components(fn_name, reduction=reduction)
    assert res == expected


def test_composite_copy():
    """Tests no infinite recursion is happening"""
    composite = Composite((MockComponent1(), MockComponent2()))
    copy(composite)
    deepcopy(composite)


def test_composite_add_component():
    composite = Composite([])
    comp1, comp2 = MockComponent1(), MockComponent2()
    composite.add_component(comp1)
    composite.add_component(comp2)

    assert composite.components == [comp2, comp1]


@pytest.mark.skip
def test_composite_unique_components():
    comp1, comp2 = MockComponent1(), MockComponent1()
    composite = Composite((comp1, comp2))

    # TODO: comp1 and comp2 are the same class but different instances
    # `set` will not consider them the same
    assert composite.components == {comp1}
