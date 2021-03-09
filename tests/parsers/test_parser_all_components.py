import pytest
from icevision.all import *


# TODO: This seems to be an uncessary test, maybe not, it tests parse_fields without
# all the loop around it
@pytest.fixture
def dummy_parser_all_components():
    class MyParser(Parser):
        def __init__(self):
            record = BaseRecord(
                (
                    FilepathRecordComponent(),
                    InstancesLabelsRecordComponent(),
                    BBoxesRecordComponent(),
                    MasksRecordComponent(),
                    KeyPointsRecordComponent(),
                    AreasRecordComponent(),
                    IsCrowdsRecordComponent(),
                )
            )
            super().__init__(record=record)

        def __iter__(self) -> Any:
            raise NotImplementedError

        def parse_fields(self, o, record: BaseRecord) -> None:
            record.set_filepath(__file__)
            record.set_img_size(ImgSize(480, 420))

            record.detect.set_class_map(ClassMap(["a"]))
            record.detect.add_labels(["a"])
            record.detect.add_bboxes([BBox.from_xyxy(1, 2, 3, 4)])
            record.detect.add_masks([MaskArray(np.zeros((1, 420, 480)))])
            record.detect.add_keypoints([KeyPoints((1, 1, 1), None)])
            record.detect.add_areas([4.2])
            record.detect.add_iscrowds([False])

    return MyParser


def test_parser_parse_fields(dummy_parser_all_components):
    parser = dummy_parser_all_components()

    record = parser.create_record()
    parser.parse_fields(None, record)

    assert record.filepath == Path(__file__)
    assert record.height == 420
    assert record.width == 480

    assert record.detect.labels == [1]
    assert record.detect.bboxes == [BBox.from_xyxy(1, 2, 3, 4)]
    assert record.detect.masks.erles == [{"size": [420, 480], "counts": b"PlT6"}]
    assert record.detect.areas == [4.2]
    assert record.detect.iscrowds == [False]
    assert record.detect.keypoints == [KeyPoints((1, 1, 1), None)]
