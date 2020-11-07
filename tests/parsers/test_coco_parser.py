import pytest
from icevision.all import *


@pytest.fixture
def coco_bbox_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=False)


@pytest.fixture
def coco_mask_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=True)


# @pytest.fixture
# def coco_keypoints_parser(coco_dir):
#     return parsers.COCOKeyPointsParser(
#         coco_dir / "keypoints_annotations.json", coco_dir / "images"
#     )


def test_keypoints_parser(coco_dir, coco_keypoints_parser):
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 2
    assert records[1].filepath == coco_dir / "images/000000404249.jpg"
    assert len(records[1].keypoints) == 1
    assert records[1].keypoints[0].n_visible_keypoints == 16
    assert records[1].keypoints[0].y.max() == 485
    assert len(records[0].keypoints) == 9


def test_bbox_parser(coco_dir, coco_bbox_parser):
    records = coco_bbox_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 5

    record = records[0]
    assert record.imageid == 0
    assert record.filepath == coco_dir / "images/000000343934.jpg"
    assert record.width == 640
    assert record.height == 480

    assert record.labels == [4]
    assert pytest.approx(record.bboxes[0].xyxy) == (175.14, 175.68, 496.2199, 415.68)
    assert record.iscrowds == [0]
    assert pytest.approx(record.areas) == [43522.805]


def test_mask_parser(coco_mask_parser):
    records = coco_mask_parser.parse(data_splitter=SingleSplitSplitter())[0]
    assert len(records) == 5

    record = records[0]
    assert record.masks.erles == [
        {
            "size": [480, 640],
            "counts": b"nWb24W10Y<=VC1`<6UC8a<T1O001O001O0cERNi7n1oG^Nm7c1iGiNT8W1hGPO"
            b"U8P1jGTOS8m0kGWOS8h0lG]OQ8b0nGAR8>lGES8:lGIS86lGMS83kG0T8OkG4T8KkG8T8HjG;"
            b"V8CiG`0V8@iGb0V8]OkGd0T8[OlGg0S8XOmGj0R8TOoGn0P8POQHR1n7lNSHV1m7gNUHZ1j"
            b"7dNWH^1h7`NYHb1f7]NZHe1e7ZN\\Hg1c7XN]Hj1b7UN^Hl1c7RN]Ho1c7PN^HQ2a7nM_HS2b"
            b"7kM^HV2b7iM_HX2`7gM`HZ2a7dM_H]2a7bM_H_2a7aM_H`2a7^M_Hc2a7]M^Hd2b7\\M^Hd2c"
            b"7[M\\Hg2c7XM]Hi2c7WM\\Hj2e7UM[Hk2e7TM[Hn2e7QM[Ho2e7QMZHP3f7oL[HQ3f7nLZHR3"
            b"f7nLYHS3g7mLYHS3g7nLXHQ3i7PMUHQ3l7oLSHQ3m7oLSHQ3m7PMQHQ3o7PMPHP3P8QMoGo2Q"
            b"8QMnGo2S8RMlGn2U8RMjGn2V8SMhGn2X8RMhGn2X8SMfGn2Z8SMeGm2[8TMdGk2]8UMbGl2_8"
            b"TM`Gl2`8UM_Gk2a8VMYGo2g8QMoFY3Q9Z110001O0000000O100000001O000O100000000O1"
            b"O1N3N1O1O1O1O1N2O1N2O2M2N2O1N2N2O1N2O1N2O2M2N2O1N3N1N2N3M2N2N3M2O1O1O2O00"
            b"001O00001O0001O00000001O00000001O00000O10000O100O2O0O10001N101N100O2O001N"
            b"101N1O101N1O2N1O2N1O101O001O0O2O001O001O001N101O001O00001O001O001O001O001"
            b"O0000O100O1O2O0O1O100O1O100O001O100O1O010O1O100O00100O1O00100N2O00@hKUF\\"
            b"4k9iKkE[4V:71O000O10001O000001O10O01O1O010O100O101N1O101N100O2O0O101O0000"
            b"1O1O2N1O2NYM\\F4b9WN]Fd0b0h0o8`NeFe0?k0j8[NnFg0:o0f8XNUGd09S1a8YNXGa09W1]"
            b"8WN]G>:[1X8UNaG=9_1T8QNjG;5d1o7lMTH;0k1i7eM`H;JQ2^9hMgFX2Q;010O10O010O1O1"
            b"00O1O1O1N2O1N2N2N101N1O2O0O2N2O0O5G=^Ob0^OXTS2",
        }
    ]
