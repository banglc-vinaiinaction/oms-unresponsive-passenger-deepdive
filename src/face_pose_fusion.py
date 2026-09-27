import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum, auto

from src.landmarks import FaceLandmarks
from src.oms_halpe import Halpe26Skeleton, compute_trunk_angle_deg, check_feet_on_dashboard
from src.dms_features import compute_ear, compute_mar, estimate_head_pose

class MultimodalState(Enum):
    NOMINAL = "NOMINAL"
    FEET_ON_DASHBOARD = "FEET_ON_DASHBOARD"
    MOTION_SICKNESS = "MOTION_SICKNESS"
    MEDICAL_INCAPACITATION = "MEDICAL_INCAPACITATION"
    HARASSMENT_VIOLENCE = "HARASSMENT_VIOLENCE"
    SENSOR_OCCLUDED_FALLBACK = "SENSOR_OCCLUDED_FALLBACK"

@dataclass
class FusedPassenger:
    passenger_id: int
    landmarks: Optional[FaceLandmarks] = None
    skeleton: Optional[Halpe26Skeleton] = None
    wrist_velocity_mps: float = 0.0
    is_face_occluded: bool = False
    is_skeleton_occluded: bool = False

def _centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

def compute_hand_to_face_dist(skeleton: Optional[Halpe26Skeleton], landmarks: Optional[FaceLandmarks]) -> float:
    """Distance from wrists to mouth/nose (detects hand covering mouth). Returns inf if sensors occluded."""
    if skeleton is None or landmarks is None:
        return float('inf')
    
    nose_pts = landmarks.nose()
    lip_pts = landmarks.outer_lips()
    face_pts = nose_pts + lip_pts
    if not face_pts:
        return float('inf')
    face_center = _centroid(face_pts)
    
    lw = skeleton.left_wrist
    rw = skeleton.right_wrist
    
    dist_l = math.hypot(lw.x - face_center[0], lw.y - face_center[1])
    dist_r = math.hypot(rw.x - face_center[0], rw.y - face_center[1])
    
    return min(dist_l, dist_r)

def _check_inter_passenger_violence(pax: FusedPassenger, all_passengers: Optional[List[FusedPassenger]]) -> bool:
    """Checks for inter-passenger violence (wrist distance < 350px and velocity > 2.5 m/s)."""
    if all_passengers is None or pax.skeleton is None:
        return False
    for other in all_passengers:
        if other.passenger_id != pax.passenger_id and other.skeleton is not None and not other.is_skeleton_occluded:
            dist_wrist = math.hypot(
                pax.skeleton.right_wrist.x - other.skeleton.left_wrist.x,
                pax.skeleton.right_wrist.y - other.skeleton.left_wrist.y
            )
            if dist_wrist < 350.0 and (pax.wrist_velocity_mps > 2.5 or other.wrist_velocity_mps > 2.5):
                return True
    return False

def evaluate_multimodal_passenger(pax: FusedPassenger, all_passengers: List[FusedPassenger] = None) -> MultimodalState:
    """
    Fuses Facial Landmarks and Body Skeleton Pose with graceful fallback behaviors:
    1. Sensor Loss / Degradation Fallback: Both sensors occluded/missing -> SENSOR_OCCLUDED_FALLBACK.
    2. Face-Occluded Fallback (Pose-Only): Trunk collapse >40° -> MEDICAL_INCAPACITATION;
       feet elevated -> FEET_ON_DASHBOARD; inter-passenger wrist dist <350 and vel >2.5 -> HARASSMENT_VIOLENCE.
    3. Skeleton-Occluded Fallback (Face-Only): EAR <0.2 and head pitch dropped (pitch <-15° or EAR <0.1) -> MEDICAL_INCAPACITATION.
    4. Full Multimodal Fusion: Violence -> Medical Incapacitation -> Motion Sickness -> Feet on Dashboard -> Nominal.
    """
    has_face = pax.landmarks is not None and not pax.is_face_occluded and len(pax.landmarks.points) >= 50
    has_skeleton = pax.skeleton is not None and not pax.is_skeleton_occluded and len(pax.skeleton.keypoints) >= 20

    # 1. Sensor Loss / Degradation Fallback
    if not has_face and not has_skeleton:
        return MultimodalState.SENSOR_OCCLUDED_FALLBACK

    # 2. Fallback Mode A: Face Occluded / Missing (Pose-Only Fallback)
    if not has_face and has_skeleton:
        trunk_angle = compute_trunk_angle_deg(pax.skeleton)
        feet_dash = check_feet_on_dashboard(pax.skeleton)

        if trunk_angle > 40.0:
            return MultimodalState.MEDICAL_INCAPACITATION
        if feet_dash:
            return MultimodalState.FEET_ON_DASHBOARD
        if _check_inter_passenger_violence(pax, all_passengers):
            return MultimodalState.HARASSMENT_VIOLENCE
        return MultimodalState.NOMINAL

    # 3. Fallback Mode B: Skeleton Occluded / Missing (Face-Only Fallback)
    if has_face and not has_skeleton:
        ear_left = compute_ear(pax.landmarks.left_eye())
        ear_right = compute_ear(pax.landmarks.right_eye())
        ear = (ear_left + ear_right) / 2.0
        yaw, pitch, roll = estimate_head_pose(pax.landmarks)

        if ear < 0.2 and (pitch < -15.0 or ear < 0.1):
            return MultimodalState.MEDICAL_INCAPACITATION
        return MultimodalState.NOMINAL

    # 4. Full Multimodal Fusion (Both sensors available)
    trunk_angle = compute_trunk_angle_deg(pax.skeleton)
    feet_dash = check_feet_on_dashboard(pax.skeleton)

    ear_left = compute_ear(pax.landmarks.left_eye())
    ear_right = compute_ear(pax.landmarks.right_eye())
    ear = (ear_left + ear_right) / 2.0
    mar = compute_mar(pax.landmarks.outer_lips(), pax.landmarks.inner_lips())
    yaw, pitch, roll = estimate_head_pose(pax.landmarks)
    hand_face_dist = compute_hand_to_face_dist(pax.skeleton, pax.landmarks)

    # 4.1 HARASSMENT_VIOLENCE
    if _check_inter_passenger_violence(pax, all_passengers):
        return MultimodalState.HARASSMENT_VIOLENCE

    # 4.2 MEDICAL_INCAPACITATION (torso collapsed + eyes closed/drooping)
    if ear < 0.2 and trunk_angle > 40.0:
        return MultimodalState.MEDICAL_INCAPACITATION

    # 4.3 MOTION_SICKNESS (evaluated before FEET_ON_DASHBOARD to avoid masking)
    if mar > 0.4 and hand_face_dist < 150.0 and abs(pitch) > 15.0:
        return MultimodalState.MOTION_SICKNESS

    # 4.4 FEET_ON_DASHBOARD (Airbag hazard)
    if feet_dash:
        return MultimodalState.FEET_ON_DASHBOARD

    return MultimodalState.NOMINAL

