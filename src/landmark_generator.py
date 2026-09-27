"""
Synthetic 50-point facial landmark sequence generator.

Produces realistic FaceLandmarks sequences for testing the DMS
perception pipeline without real IR camera data.

Scenarios:
    - Normal attentive driving (natural blinks)
    - Progressive drowsiness (increasing blink duration → microsleep)
    - Distraction (head turn / looking at phone)
    - Yawn sequences
    - Glasses condition (added noise on eye landmarks)
"""

import math
import random
from typing import List, Tuple

from src.landmarks import FaceLandmarks, NUM_LANDMARKS

# ---------------------------------------------------------------------------
# Base template: 50-point face in a 640×480 IR frame, facing forward,
# eyes open, mouth closed.
# ---------------------------------------------------------------------------

_BASE_FACE: List[Tuple[float, float]] = [
    # --- Left eyebrow (0–4) ---
    (220.0, 170.0),  # 0: outer
    (237.0, 164.0),  # 1
    (255.0, 161.0),  # 2: peak
    (273.0, 164.0),  # 3
    (288.0, 170.0),  # 4: inner

    # --- Right eyebrow (5–9) ---
    (352.0, 170.0),  # 5: inner
    (367.0, 164.0),  # 6
    (385.0, 161.0),  # 7: peak
    (403.0, 164.0),  # 8
    (420.0, 170.0),  # 9: outer

    # --- Nose bridge (10–13) ---
    (320.0, 195.0),  # 10: top (between eyes)
    (320.0, 220.0),  # 11
    (320.0, 245.0),  # 12
    (320.0, 268.0),  # 13: tip

    # --- Left eye (14–21), 8-point contour ---
    (238.0, 195.0),  # 14: outer corner
    (249.0, 186.0),  # 15: upper-outer lid
    (260.0, 184.0),  # 16: upper-mid lid
    (271.0, 186.0),  # 17: upper-inner lid
    (282.0, 195.0),  # 18: inner corner
    (271.0, 202.0),  # 19: lower-inner lid
    (260.0, 204.0),  # 20: lower-mid lid
    (249.0, 202.0),  # 21: lower-outer lid

    # --- Right eye (22–29), 8-point contour ---
    (358.0, 195.0),  # 22: inner corner
    (369.0, 186.0),  # 23: upper-inner lid
    (380.0, 184.0),  # 24: upper-mid lid
    (391.0, 186.0),  # 25: upper-outer lid
    (402.0, 195.0),  # 26: outer corner
    (391.0, 202.0),  # 27: lower-outer lid
    (380.0, 204.0),  # 28: lower-mid lid
    (369.0, 202.0),  # 29: lower-inner lid

    # --- Outer lips (30–41), 12-point contour, clockwise from left ---
    (285.0, 308.0),  # 30: left corner
    (295.0, 302.0),  # 31: upper-left-outer
    (308.0, 299.0),  # 32: upper-left
    (320.0, 297.0),  # 33: upper-center
    (332.0, 299.0),  # 34: upper-right
    (345.0, 302.0),  # 35: upper-right-outer
    (355.0, 308.0),  # 36: right corner
    (345.0, 314.0),  # 37: lower-right-outer
    (332.0, 317.0),  # 38: lower-right
    (320.0, 319.0),  # 39: lower-center
    (308.0, 317.0),  # 40: lower-left
    (295.0, 314.0),  # 41: lower-left-outer

    # --- Inner lips (42–49), 8-point contour, clockwise from left ---
    (295.0, 308.0),  # 42: left
    (308.0, 304.0),  # 43: upper-left
    (320.0, 302.0),  # 44: upper-center
    (332.0, 304.0),  # 45: upper-right
    (345.0, 308.0),  # 46: right
    (332.0, 312.0),  # 47: lower-right
    (320.0, 314.0),  # 48: lower-center
    (308.0, 312.0),  # 49: lower-left
]

assert len(_BASE_FACE) == NUM_LANDMARKS

# Reference values from the base template
_EYE_OPEN_HEIGHT = 18.0    # approx distance upper-mid to lower-mid lid
_EYE_CLOSED_GAP = 1.5      # residual gap when fully closed
_MOUTH_OPEN_YAWN = 45.0    # extra vertical spread for yawn
_EYE_Y_CENTER = 195.0      # y-level of eye corners


def _copy_base() -> List[Tuple[float, float]]:
    return [tuple(p) for p in _BASE_FACE]


def _jitter(points: List[Tuple[float, float]], sigma: float = 0.5) -> List[Tuple[float, float]]:
    """Add small Gaussian noise to simulate natural micro-movements."""
    return [
        (p[0] + random.gauss(0, sigma), p[1] + random.gauss(0, sigma))
        for p in points
    ]


def _apply_blink(
    points: List[Tuple[float, float]],
    closure_fraction: float,
) -> List[Tuple[float, float]]:
    """
    Animate eye closure by moving upper lids down and lower lids up.

    closure_fraction: 0.0 = fully open, 1.0 = fully closed.
    Applies to both left (14–21) and right (22–29) eyes.
    """
    pts = list(points)
    cf = max(0.0, min(1.0, closure_fraction))

    # Left eye upper lids (15, 16, 17) move DOWN toward eye center
    for i in [15, 16, 17]:
        base_y = _BASE_FACE[i][1]
        target_y = _EYE_Y_CENTER - _EYE_CLOSED_GAP / 2
        pts[i] = (pts[i][0], base_y + cf * (target_y - base_y))

    # Left eye lower lids (19, 20, 21) move UP toward eye center
    for i in [19, 20, 21]:
        base_y = _BASE_FACE[i][1]
        target_y = _EYE_Y_CENTER + _EYE_CLOSED_GAP / 2
        pts[i] = (pts[i][0], base_y + cf * (target_y - base_y))

    # Right eye upper lids (23, 24, 25) move DOWN
    for i in [23, 24, 25]:
        base_y = _BASE_FACE[i][1]
        target_y = _EYE_Y_CENTER - _EYE_CLOSED_GAP / 2
        pts[i] = (pts[i][0], base_y + cf * (target_y - base_y))

    # Right eye lower lids (27, 28, 29) move UP
    for i in [27, 28, 29]:
        base_y = _BASE_FACE[i][1]
        target_y = _EYE_Y_CENTER + _EYE_CLOSED_GAP / 2
        pts[i] = (pts[i][0], base_y + cf * (target_y - base_y))

    return pts


def _apply_head_turn(
    points: List[Tuple[float, float]],
    yaw_deg: float,
    pitch_deg: float,
) -> List[Tuple[float, float]]:
    """
    Simulate head pose change by shifting and scaling landmarks.

    yaw_deg:  positive = head turned to subject's left.
    pitch_deg: negative = looking down.
    """
    pts = list(points)
    cx, cy = 320.0, 240.0  # face center

    # Yaw: shift nose + compress one side
    yaw_frac = yaw_deg / 90.0  # -1 to 1
    x_shift = yaw_frac * 40.0  # nose shifts up to 40px

    for i in range(len(pts)):
        x, y = pts[i]
        dx = x - cx
        # Compress the side the face is turning away from
        scale = 1.0 - abs(yaw_frac) * 0.5  # up to 50% compression
        if (yaw_frac > 0 and dx < 0) or (yaw_frac < 0 and dx > 0):
            dx *= scale
        new_x = cx + dx + x_shift
        pts[i] = (new_x, y)

    # Pitch: simulate perspective foreshortening when looking down/up.
    # Looking down (negative pitch_deg): chin tucks in, mouth compresses
    # toward the nose, while nose stays mostly in place relative to eyes.
    # This increases nose_relative → estimator outputs negative pitch.
    if abs(pitch_deg) > 0.1:
        pitch_frac = pitch_deg / 90.0  # -0.28 for -25°
        # Mouth moves UP toward nose (positive pitch_frac = looking up = mouth drops)
        mouth_pull = pitch_frac * 60.0  # looking down → negative → mouth moves up
        for i in range(30, 50):  # all lip points
            pts[i] = (pts[i][0], pts[i][1] + mouth_pull)

    return pts


def _apply_yawn(
    points: List[Tuple[float, float]],
    openness: float,
) -> List[Tuple[float, float]]:
    """
    Animate mouth opening for yawn.
    openness: 0.0 = closed, 1.0 = fully yawning.
    """
    pts = list(points)
    o = max(0.0, min(1.0, openness))
    spread = _MOUTH_OPEN_YAWN * o

    # Upper outer lip points move up
    for i in [31, 32, 33, 34, 35]:
        pts[i] = (pts[i][0], pts[i][1] - spread * 0.3)
    # Lower outer lip points move down
    for i in [37, 38, 39, 40, 41]:
        pts[i] = (pts[i][0], pts[i][1] + spread * 0.7)
    # Inner lips
    for i in [43, 44, 45]:
        pts[i] = (pts[i][0], pts[i][1] - spread * 0.25)
    for i in [47, 48, 49]:
        pts[i] = (pts[i][0], pts[i][1] + spread * 0.65)

    return pts


# ---------------------------------------------------------------------------
# Public generators
# ---------------------------------------------------------------------------

def generate_normal_driving(
    duration_s: float = 10.0,
    fps: int = 60,
    seed: int = 42,
) -> List[FaceLandmarks]:
    """
    Normal attentive driving: eyes open, looking forward,
    natural blinks every 3–5 s (duration ~200 ms).
    """
    random.seed(seed)
    frames: List[FaceLandmarks] = []
    total_frames = int(duration_s * fps)
    dt = 1.0 / fps

    # Pre-schedule blink events
    blink_times = []
    t = random.uniform(2.0, 4.0)
    while t < duration_s:
        blink_times.append(t)
        t += random.uniform(3.0, 5.0)

    blink_duration = 0.20  # 200ms normal blink

    for f in range(total_frames):
        t = f * dt
        pts = _copy_base()

        # Check if we're in a blink
        closure = 0.0
        for bt in blink_times:
            if bt <= t < bt + blink_duration:
                # Smooth blink curve (sine half-wave)
                phase = (t - bt) / blink_duration
                closure = math.sin(math.pi * phase)
                break

        pts = _apply_blink(pts, closure)
        pts = _jitter(pts, sigma=0.4)

        frames.append(FaceLandmarks(
            points=pts,
            timestamp=t,
            glasses=False,
        ))

    return frames


def generate_drowsy_sequence(
    duration_s: float = 15.0,
    fps: int = 60,
    seed: int = 123,
) -> List[FaceLandmarks]:
    """
    Progressive drowsiness: blinks get longer, then sustained closure
    (microsleep) in the final third.
    """
    random.seed(seed)
    frames: List[FaceLandmarks] = []
    total_frames = int(duration_s * fps)
    dt = 1.0 / fps

    # Phase 1 (0–40%): normal blinks
    # Phase 2 (40–70%): slow, long blinks
    # Phase 3 (70–100%): microsleep (sustained closure)
    phase1_end = duration_s * 0.40
    phase2_end = duration_s * 0.70

    # Schedule blinks
    blinks = []  # (start_time, duration)
    t = random.uniform(1.5, 3.0)
    while t < phase2_end:
        if t < phase1_end:
            bd = random.uniform(0.15, 0.25)
            gap = random.uniform(3.0, 5.0)
        else:
            bd = random.uniform(0.5, 0.9)
            gap = random.uniform(2.0, 3.5)
        blinks.append((t, bd))
        t += bd + gap

    for f in range(total_frames):
        t = f * dt
        pts = _copy_base()

        closure = 0.0
        if t >= phase2_end:
            # Microsleep: eyes stay closed
            closure = 1.0
        else:
            for bt, bd in blinks:
                if bt <= t < bt + bd:
                    phase = (t - bt) / bd
                    closure = math.sin(math.pi * phase)
                    break

        # Add increasing baseline droop in phase 2
        if phase1_end <= t < phase2_end:
            droop_progress = (t - phase1_end) / (phase2_end - phase1_end)
            closure = max(closure, droop_progress * 0.15)

        pts = _apply_blink(pts, closure)
        pts = _jitter(pts, sigma=0.5)

        frames.append(FaceLandmarks(
            points=pts,
            timestamp=t,
            glasses=False,
        ))

    return frames


def generate_distraction_sequence(
    duration_s: float = 10.0,
    fps: int = 60,
    seed: int = 456,
) -> List[FaceLandmarks]:
    """
    Distraction: head turns to look at phone (pitch down, slight yaw)
    starting at ~30% into the sequence.
    """
    random.seed(seed)
    frames: List[FaceLandmarks] = []
    total_frames = int(duration_s * fps)
    dt = 1.0 / fps

    distraction_start = duration_s * 0.30
    ramp_duration = 1.0  # 1s to turn head

    for f in range(total_frames):
        t = f * dt
        pts = _copy_base()

        if t >= distraction_start:
            progress = min(1.0, (t - distraction_start) / ramp_duration)
            yaw = progress * 10.0   # slight left turn
            pitch = progress * -25.0  # looking down
            pts = _apply_head_turn(pts, yaw, pitch)

        pts = _jitter(pts, sigma=0.4)

        frames.append(FaceLandmarks(
            points=pts,
            timestamp=t,
            glasses=False,
        ))

    return frames


def generate_yawn_sequence(
    duration_s: float = 10.0,
    fps: int = 60,
    seed: int = 789,
) -> List[FaceLandmarks]:
    """
    Normal driving with periodic yawns (MAR spikes, ~2s each).
    """
    random.seed(seed)
    frames: List[FaceLandmarks] = []
    total_frames = int(duration_s * fps)
    dt = 1.0 / fps

    # Schedule yawns
    yawn_times = [3.0, 7.0]
    yawn_duration = 2.0

    for f in range(total_frames):
        t = f * dt
        pts = _copy_base()

        # Natural blinks
        blink_closure = 0.0

        # Yawn
        yawn_openness = 0.0
        for yt in yawn_times:
            if yt <= t < yt + yawn_duration:
                phase = (t - yt) / yawn_duration
                yawn_openness = math.sin(math.pi * phase)
                break

        pts = _apply_blink(pts, blink_closure)
        pts = _apply_yawn(pts, yawn_openness)
        pts = _jitter(pts, sigma=0.4)

        frames.append(FaceLandmarks(
            points=pts,
            timestamp=t,
            glasses=False,
        ))

    return frames


def generate_glasses_sequence(
    duration_s: float = 10.0,
    fps: int = 60,
    seed: int = 999,
) -> List[FaceLandmarks]:
    """
    Same as normal driving but with glasses=True:
    - Higher noise on eye landmarks (gọng kính occlusion)
    - Occasional IR reflection spikes (false low EAR)
    """
    random.seed(seed)
    base_frames = generate_normal_driving(duration_s, fps, seed=seed + 1)

    # Add glasses artifacts
    reflection_times = [2.5, 5.8, 8.1]  # IR reflection events
    reflection_duration = 0.08  # 80ms flash

    for frame in base_frames:
        frame.glasses = True

        # Extra noise on eye landmarks
        new_pts = list(frame.points)
        for i in list(range(14, 22)) + list(range(22, 30)):
            x, y = new_pts[i]
            new_pts[i] = (
                x + random.gauss(0, 1.5),  # 3x normal noise
                y + random.gauss(0, 1.5),
            )

        # IR reflection: collapse upper lid toward lower on left eye
        for rt in reflection_times:
            if rt <= frame.timestamp < rt + reflection_duration:
                for i in [15, 16, 17]:  # left upper lid
                    x, y = new_pts[i]
                    new_pts[i] = (x, y + 12.0)  # Push down → false closure
                frame.occluded = list(frame.occluded)
                for i in [15, 16, 17]:
                    frame.occluded[i] = True
                break

        frame.points = new_pts

    return base_frames
