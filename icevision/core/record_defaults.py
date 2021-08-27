__all__ = [
    "ObjectDetectionRecord",
    "InstanceSegmentationRecord",
    "KeypointsRecord",
    "RadiographicObjectDetectionRecord",
    "RadiographicInstanceSegmentationRecord",
    "RadiographicKeypointsRecord",
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


def RadiographicObjectDetectionRecord():
    return BaseRecord(
        (
            RadiographicRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )


def RadiographicInstanceSegmentationRecord():
    return BaseRecord(
        (
            RadiographicRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            MasksRecordComponent(),
        )
    )


def RadiographicKeypointsRecord():
    return BaseRecord(
        (
            RadiographicRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            KeyPointsRecordComponent(),
        )
    )
