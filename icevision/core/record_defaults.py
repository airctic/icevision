__all__ = [
    "ObjectDetectionRecord",
    "InstanceSegmentationRecord",
    "KeypointsRecord",
    "GrayScaleObjectDetectionRecord",
    "GrayScaleInstanceSegmentationRecord",
    "GrayScaleKeypointsRecord",
]

from icevision.core.record import *
from icevision.core.record_components import *


def ObjectDetectionRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )


def InstanceSegmentationRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            MasksRecordComponent(),
        )
    )


def KeypointsRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            KeyPointsRecordComponent(),
        )
    )


def GrayScaleObjectDetectionRecord():
    return BaseRecord(
        (
            GrayScaleRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )


def GrayScaleInstanceSegmentationRecord():
    return BaseRecord(
        (
            GrayScaleRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            MasksRecordComponent(),
        )
    )


def GrayScaleKeypointsRecord():
    return BaseRecord(
        (
            GrayScaleRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            KeyPointsRecordComponent(),
        )
    )
