from icevision.all import *


def test_match_components(coco_bbox_parser):
    parser_components = coco_bbox_parser.components

    record_components = component_registry.match_components(
        RecordComponent, parser_components
    )

    expected_record_components = (
        ImageidRecordComponent,
        ClassMapRecordComponent,
        FilepathRecordComponent,
        SizeRecordComponent,
        LabelsRecordComponent,
        BBoxesRecordComponent,
        AreasRecordComponent,
        IsCrowdsRecordComponent,
    )

    assert set(record_components) == set(expected_record_components)
