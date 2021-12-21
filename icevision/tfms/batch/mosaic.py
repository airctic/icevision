from icevision import tfms
from icevision.imports import *
from icevision.core import *
from icevision.tfms.batch.batch_transform import BatchTransform


class Mosaic(BatchTransform):
    def __init__(self, n_imgs=4, bbox_safe=True):
        self.n_imgs = n_imgs
        self.bbox_safe = bbox_safe

    def create_tfms(self, main_record: BaseRecord):
        positions = [
            tfms.A.transforms.PadIfNeeded.PositionType.TOP_LEFT,
            tfms.A.transforms.PadIfNeeded.PositionType.TOP_RIGHT,
            tfms.A.transforms.PadIfNeeded.PositionType.BOTTOM_LEFT,
            tfms.A.transforms.PadIfNeeded.PositionType.BOTTOM_RIGHT,
        ]
        h = main_record.img_size.height
        w = main_record.img_size.width
        rw, rh = (1 + np.random.random_sample(2)) / 3
        pw, ph = int(rw * w), int(rh * h)

        crop_boundaries = [
            [0, 0, pw, ph],
            [pw, 0, w, ph],
            [0, ph, pw, h],
            [pw, ph, w, h],
        ]

        mosaic_tfms = [
            tfms.A.Adapter(
                [
                    tfms.A.RandomSizedBBoxSafeCrop(y_max - y_min, x_max - x_min)
                    if self.bbox_safe
                    else tfms.A.Crop(x_min, y_min, x_max, y_max),
                    tfms.A.PadIfNeeded(h, w, position=position, border_mode=0),
                ]
            )
            for (x_min, y_min, x_max, y_max), position in zip(
                crop_boundaries, positions
            )
        ]

        return mosaic_tfms

    def apply(self, records: List[BaseRecord]) -> List[BaseRecord]:
        transformed_records = []
        for i, current_record in enumerate(records):
            other_records = records[:i] + records[i + 1 :]

            n = self.n_imgs - 1  # n of images to draw
            mosaic_records = np.random.choice(other_records, size=n, replace=False)

            # cannot edit record.img directly cause then we end up with nested mosaics
            transformed_records.append(self.make_mosaic(current_record, mosaic_records))

        return transformed_records

    def make_mosaic(
        self, main_record: BaseRecord, mosaic_records: List[BaseRecord]
    ) -> BaseRecord:
        mosaic_tfms = self.create_tfms(main_record)
        main_record_copy = deepcopy(main_record)
        canvas = np.zeros_like(main_record.img)
        labels = []
        bboxes = []
        # apply crops and padding
        for record, tfm in zip([main_record, *mosaic_records], mosaic_tfms):
            record_copy = deepcopy(record)
            record_copy = tfm(record_copy)
            canvas += record_copy.img

            labels.extend(record_copy.detection.labels)
            bboxes.extend(record_copy.detection.bboxes)

        # compile transformed mosaic record
        main_record_copy.set_img(canvas)
        main_record_copy.detection.set_bboxes(bboxes)
        main_record_copy.detection.set_labels(labels)
        return main_record_copy
