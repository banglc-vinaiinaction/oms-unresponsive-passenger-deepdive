"""
Rule-based DMS classifier: continuous features → discrete enums.

Maps EAR, MAR, PERCLOS, and head pose to GazeZone, AttentionState,
and DrowsinessLevel consumed by the DriverInterventionStateMachine.

This is the 'đặc trưng thủ công (rule-based)' path from slide 21.
"""

from dataclasses import dataclass
from typing import Dict, Any

from src.states import AttentionState, GazeZone, DrowsinessLevel
from src.dms_features import (
    compute_ear, compute_mar, estimate_head_pose, PERCLOSTracker,
)
from src.landmarks import FaceLandmarks


@dataclass
class DMSOutput:
    """All DMS perception outputs for a single frame."""
    gaze_zone: GazeZone
    attention_state: AttentionState
    drowsiness_level: DrowsinessLevel
    ear_left: float
    ear_right: float
    ear_avg: float
    mar: float
    head_yaw: float
    head_pitch: float
    head_roll: float
    perclos: float
    is_yawning: bool
    eyes_closed: bool


class DMSClassifier:
    """
    Threshold-based classifier translating raw geometric features
    into the enums expected by the escalation engine.
    """

    # --- Eye closure ---
    EAR_CLOSED_THRESHOLD = 0.20

    # --- Yawn detection ---
    MAR_YAWN_THRESHOLD = 0.50
    YAWN_MIN_DURATION_S = 1.0

    # --- PERCLOS → DrowsinessLevel ---
    PERCLOS_MILD = 0.10
    PERCLOS_MODERATE = 0.15
    PERCLOS_SEVERE = 0.25
    PERCLOS_SLEEP_ONSET = 0.40

    # --- Head pose → GazeZone ---
    YAW_MIRROR_MIN = 20.0       # degrees
    YAW_MIRROR_MAX = 45.0
    YAW_AWAY = 45.0
    PITCH_DOWN_PHONE = -15.0    # looking down at phone
    PITCH_DOWN_CLUSTER = -8.0   # glancing at cluster

    # --- Eye-closure duration → microsleep ---
    MICROSLEEP_THRESHOLD_S = 0.5

    def __init__(self, perclos_window_s: float = 60.0):
        self.perclos_tracker = PERCLOSTracker(
            window_seconds=perclos_window_s,
            ear_threshold=self.EAR_CLOSED_THRESHOLD,
        )
        self._yawn_start: float | None = None
        self._eye_closed_start: float | None = None
        self._current_time = 0.0
        self._consecutive_yawns = 0
        self._last_yawn_end = 0.0

    def classify(self, landmarks: FaceLandmarks, dt: float) -> DMSOutput:
        """
        Classify a single frame of landmarks.

        Args:
            landmarks: 50-point FaceLandmarks for this frame.
            dt: time delta since last frame (seconds).

        Returns:
            DMSOutput with all features and classified enums.
        """
        self._current_time += dt

        # --- Compute raw features ---
        ear_l = compute_ear(landmarks.left_eye())
        ear_r = compute_ear(landmarks.right_eye())
        ear_avg = (ear_l + ear_r) / 2.0

        mar = compute_mar(landmarks.outer_lips(), landmarks.inner_lips())
        yaw, pitch, roll = estimate_head_pose(landmarks)

        perclos = self.perclos_tracker.update(self._current_time, ear_avg)

        # --- Eye closure tracking ---
        eyes_closed = ear_avg < self.EAR_CLOSED_THRESHOLD
        if eyes_closed:
            if self._eye_closed_start is None:
                self._eye_closed_start = self._current_time
        else:
            self._eye_closed_start = None

        eye_closed_duration = (
            (self._current_time - self._eye_closed_start)
            if self._eye_closed_start is not None
            else 0.0
        )
        is_microsleep = eye_closed_duration >= self.MICROSLEEP_THRESHOLD_S

        # --- Yawn tracking ---
        is_yawning = mar >= self.MAR_YAWN_THRESHOLD
        if is_yawning:
            if self._yawn_start is None:
                self._yawn_start = self._current_time
        else:
            if self._yawn_start is not None:
                yawn_dur = self._current_time - self._yawn_start
                if yawn_dur >= self.YAWN_MIN_DURATION_S:
                    self._consecutive_yawns += 1
                    self._last_yawn_end = self._current_time
                self._yawn_start = None
        # Decay yawn count after 5 minutes
        if self._current_time - self._last_yawn_end > 300.0:
            self._consecutive_yawns = 0

        sustained_yawn = (
            is_yawning
            and self._yawn_start is not None
            and (self._current_time - self._yawn_start) >= self.YAWN_MIN_DURATION_S
        )

        # --- Classify GazeZone ---
        gaze = self._classify_gaze(yaw, pitch, eyes_closed)

        # --- Classify DrowsinessLevel ---
        drowsiness = self._classify_drowsiness(
            perclos, sustained_yawn, is_microsleep
        )

        # --- Classify AttentionState ---
        attention = self._classify_attention(gaze, eyes_closed, is_microsleep)

        return DMSOutput(
            gaze_zone=gaze,
            attention_state=attention,
            drowsiness_level=drowsiness,
            ear_left=ear_l,
            ear_right=ear_r,
            ear_avg=ear_avg,
            mar=mar,
            head_yaw=yaw,
            head_pitch=pitch,
            head_roll=roll,
            perclos=perclos,
            is_yawning=is_yawning,
            eyes_closed=eyes_closed,
        )

    # ------------------------------------------------------------------
    # Internal classification helpers
    # ------------------------------------------------------------------

    def _classify_gaze(
        self, yaw: float, pitch: float, eyes_closed: bool
    ) -> GazeZone:
        if eyes_closed:
            return GazeZone.EYES_CLOSED

        abs_yaw = abs(yaw)

        # Pitch-based (looking down)
        if pitch < self.PITCH_DOWN_PHONE:
            return GazeZone.PHONE_DOWN
        if pitch < self.PITCH_DOWN_CLUSTER:
            return GazeZone.CLUSTER

        # Yaw-based (looking sideways)
        if abs_yaw >= self.YAW_AWAY:
            return GazeZone.UNKNOWN  # Extreme head turn
        if self.YAW_MIRROR_MIN <= abs_yaw < self.YAW_MIRROR_MAX:
            return GazeZone.LEFT_MIRROR if yaw > 0 else GazeZone.RIGHT_MIRROR

        return GazeZone.ROAD_AHEAD

    def _classify_drowsiness(
        self,
        perclos: float,
        sustained_yawn: bool,
        is_microsleep: bool,
    ) -> DrowsinessLevel:
        if is_microsleep:
            return DrowsinessLevel.SLEEP_ONSET

        # PERCLOS-based
        if perclos >= self.PERCLOS_SLEEP_ONSET:
            level = DrowsinessLevel.SLEEP_ONSET
        elif perclos >= self.PERCLOS_SEVERE:
            level = DrowsinessLevel.SEVERE_DROWSY
        elif perclos >= self.PERCLOS_MODERATE:
            level = DrowsinessLevel.MODERATE_DROWSY
        elif perclos >= self.PERCLOS_MILD:
            level = DrowsinessLevel.MILD_DROWSY
        else:
            level = DrowsinessLevel.ALERT

        # Yawn boost: bump one level if actively yawning
        if sustained_yawn and level < DrowsinessLevel.SEVERE_DROWSY:
            level = DrowsinessLevel(min(level.value + 1, DrowsinessLevel.SLEEP_ONSET.value))

        # Repeated yawns boost
        if self._consecutive_yawns >= 3 and level < DrowsinessLevel.MODERATE_DROWSY:
            level = DrowsinessLevel.MODERATE_DROWSY

        return level

    def _classify_attention(
        self,
        gaze: GazeZone,
        eyes_closed: bool,
        is_microsleep: bool,
    ) -> AttentionState:
        if is_microsleep:
            return AttentionState.MICROSLEEP
        if eyes_closed:
            return AttentionState.DROWSY

        if gaze in (GazeZone.ROAD_AHEAD, GazeZone.LEFT_MIRROR,
                     GazeZone.RIGHT_MIRROR, GazeZone.CLUSTER):
            return AttentionState.ATTENTIVE
        if gaze == GazeZone.PHONE_DOWN:
            return AttentionState.DISTRACTED_HIGH
        if gaze == GazeZone.INFOTAINMENT:
            return AttentionState.DISTRACTED_LOW
        # UNKNOWN or other off-road gaze
        return AttentionState.DISTRACTED_HIGH
