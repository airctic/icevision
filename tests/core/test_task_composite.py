# from icevision.all import *

# first_task = tasks.Task("first")
# second_task = tasks.Task("second")

# record = BaseRecord(
#     (
#         FilepathRecordComponent(),
#         InstancesLabelsRecordComponent(task=first_task),
#         BBoxesRecordComponent(task=first_task),
#         InstancesLabelsRecordComponent(task=second_task),
#         BBoxesRecordComponent(task=second_task),
#     )
# )

# record.builder_template()

# [
#     "record.set_img_size(<ImgSize>)",
#     "record.set_filepath(<Union[str, Path]>)",
#     "record.first.add_labels_names(<Sequence[Hashable]>)",
#     "record.first.add_bboxes(<Sequence[BBox]>)",
#     "record.second.add_labels_names(<Sequence[Hashable]>)",
#     "record.second.add_bboxes(<Sequence[BBox]>)",
# ]
