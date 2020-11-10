__all__ = ["KeyPoints"]

from icevision.imports import *
from .exceptions import *


class KeyPoints:
    """
    docs
    """

    def __init__(self, keypoints: Union[List[int], np.array]):
        self.keypoints = np.array([int(round(k)) for k in keypoints])
        self.x = self.keypoints[0::3]
        self.y = self.keypoints[1::3]
        self.visible = self.keypoints[2::3]
        self.xy = [(x, y) for x, y in zip(self.x, self.y)]
        self.n_visible_keypoints = (self.visible > 0).sum()

        self.human_kps = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.human_conns = np.array(
            [
                [15, 13],
                [13, 11],
                [16, 14],
                [14, 12],
                [11, 12],
                [5, 11],
                [6, 12],
                [5, 6],
                [5, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [1, 2],
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
            ]
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} " f"(N keypoints: {self.n_visible_keypoints})>"
        )

    @classmethod
    def from_xyv(cls, keypoints):
        return cls(keypoints)
