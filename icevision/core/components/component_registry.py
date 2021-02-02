__all__ = ["component_registry"]

from icevision.imports import *


class _ComponentRegistry:
    def __init__(self):
        self.components = {}
        self.component2name = {}

    def new_component_registry(self, name):
        if name in self.components:
            raise ValueError("{name} is already registered")

        self.components[name] = []
        return partial(self.register_component, name=name)

    def register_component(self, component, name):
        self.components[name].append(component)
        self.component2name[component] = name
        return component

    def get_components_groups(self, components):
        names = []
        for component in components:
            try:
                names.append(self.component2name[component])
            except KeyError:
                pass
        return names

    def match_components(self, base_cls, components):
        """Matches components with the same type but for base_cls."""
        names = self.get_components_groups(components)
        return [
            comp
            for name in names
            for comp in self.components[name]
            if issubclass(comp, base_cls)
        ]


component_registry = _ComponentRegistry()
