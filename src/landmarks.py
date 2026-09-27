"""
50-point facial landmark schema for DMS (Driver Monitoring System).

Layout (slide 19, Day 10):
    Left eyebrow:  5 points  (indices  0– 4)
    Right eyebrow: 5 points  (indices  5– 9)
    Nose bridge:   4 points  (indices 10–13)
    Left eye:      8 points  (indices 14–21)
    Right eye:     8 points  (indices 22–29)
    Outer lips:   12 points  (indices 30–41)
    Inner lips:    8 points  (indices 42–49)
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import enum

NUM_LANDMARKS = 50


class LandmarkRegion(enum.Enum):
    LEFT_EYEBROW = "left_eyebrow"
    RIGHT_EYEBROW = "right_eyebrow"
    NOSE_BRIDGE = "nose_bridge"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    OUTER_LIPS = "outer_lips"
    INNER_LIPS = "inner_lips"


REGION_INDICES = {
    LandmarkRegion.LEFT_EYEBROW: range(0, 5),
    LandmarkRegion.RIGHT_EYEBROW: range(5, 10),
    LandmarkRegion.NOSE_BRIDGE: range(10, 14),
    LandmarkRegion.LEFT_EYE: range(14, 22),
    LandmarkRegion.RIGHT_EYE: range(22, 30),
    LandmarkRegion.OUTER_LIPS: range(30, 42),
    LandmarkRegion.INNER_LIPS: range(42, 50),
}

# 8-point eye topology (used by EAR):
#   0: outer corner
#   1: upper-outer lid
#   2: upper-mid lid
#   3: upper-inner lid
#   4: inner corner
#   5: lower-inner lid
#   6: lower-mid lid
#   7: lower-outer lid
EYE_OUTER_CORNER = 0
EYE_UPPER_OUTER = 1
EYE_UPPER_MID = 2
EYE_UPPER_INNER = 3
EYE_INNER_CORNER = 4
EYE_LOWER_INNER = 5
EYE_LOWER_MID = 6
EYE_LOWER_OUTER = 7

# 12-point outer lip topology (used by MAR):
#   0: left corner
#   3: upper center
#   6: right corner
#   9: lower center
OUTER_LIP_LEFT = 0
OUTER_LIP_UPPER_CENTER = 3
OUTER_LIP_RIGHT = 6
OUTER_LIP_LOWER_CENTER = 9


@dataclass
class FaceLandmarks:
    """50 facial landmark coordinates for a single frame."""
    points: List[Tuple[float, float]]   # [(x, y), ...] length 50
    occluded: List[bool] = field(default_factory=lambda: [False] * NUM_LANDMARKS)
    timestamp: float = 0.0
    glasses: bool = False

    def __post_init__(self):
        if len(self.points) != NUM_LANDMARKS:
            raise ValueError(
                f"Expected {NUM_LANDMARKS} landmarks, got {len(self.points)}"
            )
        if len(self.occluded) != NUM_LANDMARKS:
            self.occluded = [False] * NUM_LANDMARKS

    def get_region(self, region: LandmarkRegion) -> List[Tuple[float, float]]:
        """Return points for a specific facial region."""
        return [self.points[i] for i in REGION_INDICES[region]]

    def left_eye(self) -> List[Tuple[float, float]]:
        return self.get_region(LandmarkRegion.LEFT_EYE)

    def right_eye(self) -> List[Tuple[float, float]]:
        return self.get_region(LandmarkRegion.RIGHT_EYE)

    def outer_lips(self) -> List[Tuple[float, float]]:
        return self.get_region(LandmarkRegion.OUTER_LIPS)

    def inner_lips(self) -> List[Tuple[float, float]]:
        return self.get_region(LandmarkRegion.INNER_LIPS)

    def nose(self) -> List[Tuple[float, float]]:
        return self.get_region(LandmarkRegion.NOSE_BRIDGE)
