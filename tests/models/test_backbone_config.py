import pytest

from icevision.models.backbone_config import BackboneConfig


def test_backbone_config_is_abstract_base_class():
    with pytest.raises(TypeError):
        _ = BackboneConfig()
