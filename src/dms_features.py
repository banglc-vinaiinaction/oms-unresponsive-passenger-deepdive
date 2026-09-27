"""
DMS geometric feature extraction from 50 facial landmarks.

Implements:
    - EAR  (Eye Aspect Ratio)     — Soukupová & Čech, 2016
    - MAR  (Mouth Aspect Ratio)   — same principle applied to lips
    - Head pose (yaw/pitch/roll)  — simplified 2D geometric estimation
    - PERCLOS tracker             — Wierwille et al., 1994
"""

import math
from typing import List, Tuple
from collections import deque

from src.landmarks import (
    FaceLandmarks, LandmarkRegion, REGION_INDICES,
    EYE_OUTER_CORNER, EYE_UPPER_OUTER, EYE_UPPER_MID, EYE_UPPER_INNER,
    EYE_INNER_CORNER, EYE_LOWER_INNER, EYE_LOWER_MID, EYE_LOWER_OUTER,
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = len(pts)
    if n == 0:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


# ---------------------------------------------------------------------------
# EAR — Eye Aspect Ratio
# ---------------------------------------------------------------------------

def compute_ear(eye_points: List[Tuple[float, float]]) -> float:
    """
    Eye Aspect Ratio for an 8-point eye contour.

    8-point layout (see landmarks.py):
        0: outer corner
        1: upper-outer lid
        2: upper-mid lid
        3: upper-inner lid
        4: inner corner
        5: lower-inner lid
        6: lower-mid lid
        7: lower-outer lid

    EAR = (‖p1−p7‖ + ‖p2−p6‖ + ‖p3−p5‖) / (3 × ‖p0−p4‖)

    Open eye ≈ 0.25–0.35, closed eye ≈ 0.02–0.08.
    """
    if len(eye_points) != 8:
        raise ValueError(f"Expected 8 eye points, got {len(eye_points)}")

    p = eye_points
    vertical_sum = (
        _dist(p[EYE_UPPER_OUTER], p[EYE_LOWER_OUTER])
        + _dist(p[EYE_UPPER_MID], p[EYE_LOWER_MID])
        + _dist(p[EYE_UPPER_INNER], p[EYE_LOWER_INNER])
    )
    horizontal = _dist(p[EYE_OUTER_CORNER], p[EYE_INNER_CORNER])
    if horizontal < 1e-6:
        return 0.0
    return vertical_sum / (3.0 * horizontal)


# ---------------------------------------------------------------------------
# MAR — Mouth Aspect Ratio
# ---------------------------------------------------------------------------

def compute_mar(
    outer_lip_points: List[Tuple[float, float]],
    inner_lip_points: List[Tuple[float, float]],
) -> float:
    """
    Mouth Aspect Ratio — detects yawning.

    Uses 3 vertical outer-lip pair distances divided by horizontal width.
    Outer lip layout (12 points, clockwise from left corner):
        0: left corner,  3: upper-center,  6: right corner,  9: lower-center
        Pairs: (2,10), (3,9), (4,8)

    Closed mouth ≈ 0.15–0.25,  yawn ≈ 0.55+.
    """
    if len(outer_lip_points) != 12:
        raise ValueError(f"Expected 12 outer lip points, got {len(outer_lip_points)}")

    o = outer_lip_points
    vertical_sum = (
        _dist(o[2], o[10])
        + _dist(o[3], o[9])
        + _dist(o[4], o[8])
    )
    horizontal = _dist(o[0], o[6])
    if horizontal < 1e-6:
        return 0.0
    return vertical_sum / (3.0 * horizontal)


# ---------------------------------------------------------------------------
# Head Pose (2D geometric approximation)
# ---------------------------------------------------------------------------

def estimate_head_pose(
    landmarks: FaceLandmarks,
) -> Tuple[float, float, float]:
    """
    Estimates (yaw, pitch, roll) in degrees from 2D landmark geometry.

    Yaw:   asymmetry of nose-to-eye-center horizontal distances.
    Pitch: nose vertical position relative to eye→mouth baseline.
    Roll:  angle of the inter-eye line.

    This is a coarse 2D proxy — no camera intrinsics needed.
    Returns (yaw_deg, pitch_deg, roll_deg).
    """
    left_eye_pts = landmarks.left_eye()
    right_eye_pts = landmarks.right_eye()
    nose_pts = landmarks.nose()
    outer_lips = landmarks.outer_lips()

    left_eye_center = _centroid(left_eye_pts)
    right_eye_center = _centroid(right_eye_pts)
    nose_tip = nose_pts[-1]  # index 13 = bottom of bridge (tip)
    mouth_center = (
        (outer_lips[3][0] + outer_lips[9][0]) / 2.0,
        (outer_lips[3][1] + outer_lips[9][1]) / 2.0,
    )

    # ---- Yaw ----
    # Horizontal distance from nose tip to each eye center
    d_left = abs(nose_tip[0] - left_eye_center[0])
    d_right = abs(nose_tip[0] - right_eye_center[0])
    denom = d_left + d_right
    if denom < 1e-6:
        yaw_deg = 0.0
    else:
        # Positive = head turned to subject's left (nose closer to right eye in image)
        yaw_ratio = (d_right - d_left) / denom
        yaw_deg = yaw_ratio * 90.0

    # ---- Roll ----
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    roll_deg = math.degrees(math.atan2(dy, dx)) if abs(dx) > 1e-6 else 0.0

    # ---- Pitch ----
    eye_mid_y = (left_eye_center[1] + right_eye_center[1]) / 2.0
    face_height = mouth_center[1] - eye_mid_y
    if abs(face_height) < 1e-6:
        pitch_deg = 0.0
    else:
        nose_relative = (nose_tip[1] - eye_mid_y) / face_height
        # Baseline when facing forward: nose is ~65% of the way from eyes to mouth
        BASELINE = 0.65
        pitch_deg = (nose_relative - BASELINE) * -130.0

    return (yaw_deg, pitch_deg, roll_deg)


# ---------------------------------------------------------------------------
# PERCLOS — Percentage of Eyelid Closure
# ---------------------------------------------------------------------------

class PERCLOSTracker:
    """
    Tracks the percentage of time eyes are closed (EAR < threshold)
    over a sliding time window.

    PERCLOS > 15% is a widely used drowsiness indicator
    (Wierwille et al., FHWA, 1994).
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        ear_threshold: float = 0.20,
    ):
        self.window_seconds = window_seconds
        self.ear_threshold = ear_threshold
        self._history: deque = deque()  # (timestamp, is_closed: bool)

    def update(self, timestamp: float, ear: float) -> float:
        """
        Record an EAR sample and return current PERCLOS (0.0–1.0).

        Args:
            timestamp: absolute time in seconds.
            ear: average EAR of both eyes for this frame.

        Returns:
            Fraction of samples in the window where eyes were closed.
        """
        is_closed = ear < self.ear_threshold
        self._history.append((timestamp, is_closed))

        # Evict samples outside the window
        cutoff = timestamp - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if not self._history:
            return 0.0

        closed_count = sum(1 for _, c in self._history if c)
        return closed_count / len(self._history)

    def reset(self):
        self._history.clear()
