import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
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

@dataclass
class FusedPassenger:
    passenger_id: int
    landmarks: FaceLandmarks
    skeleton: Halpe26Skeleton
    wrist_velocity_mps: float = 0.0

def _centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not pts:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

def compute_hand_to_face_dist(skeleton: Halpe26Skeleton, landmarks: FaceLandmarks) -> float:
    """Distance from wrists to mouth/nose (detects hand covering mouth)."""
    nose_pts = landmarks.nose()
    lip_pts = landmarks.outer_lips()
    face_pts = nose_pts + lip_pts
    face_center = _centroid(face_pts)
    
    lw = skeleton.left_wrist
    rw = skeleton.right_wrist
    
    dist_l = math.hypot(lw.x - face_center[0], lw.y - face_center[1])
    dist_r = math.hypot(rw.x - face_center[0], rw.y - face_center[1])
    
    return min(dist_l, dist_r)

def evaluate_multimodal_passenger(pax: FusedPassenger, all_passengers: List[FusedPassenger] = None) -> MultimodalState:
    """Fuses EAR, MAR, Head Pose, Trunk Angle, Foot elevation, and Hand-to-face distance."""
    ear_left = compute_ear(pax.landmarks.left_eye())
    ear_right = compute_ear(pax.landmarks.right_eye())
    ear = (ear_left + ear_right) / 2.0
    
    mar = compute_mar(pax.landmarks.outer_lips(), pax.landmarks.inner_lips())
    yaw, pitch, roll = estimate_head_pose(pax.landmarks)
    trunk_angle = compute_trunk_angle_deg(pax.skeleton)
    feet_dash = check_feet_on_dashboard(pax.skeleton)
    hand_face_dist = compute_hand_to_face_dist(pax.skeleton, pax.landmarks)
    
    # HARASSMENT_VIOLENCE (Inter-passenger wrist distance < 0.35m + high velocity)
    if all_passengers is not None:
        for other in all_passengers:
            if other.passenger_id != pax.passenger_id:
                # Inter-passenger distance check. Assume ~1000 pixels = 1 meter, so 350 pixels = 0.35m
                dist_wrist = math.hypot(pax.skeleton.right_wrist.x - other.skeleton.left_wrist.x,
                                        pax.skeleton.right_wrist.y - other.skeleton.left_wrist.y)
                if dist_wrist < 350.0 and (pax.wrist_velocity_mps > 2.5 or other.wrist_velocity_mps > 2.5):
                    return MultimodalState.HARASSMENT_VIOLENCE
                    
    # MEDICAL_INCAPACITATION (Zero EAR / eyes closed + torso collapsed)
    if ear < 0.2 and trunk_angle > 40.0:
        return MultimodalState.MEDICAL_INCAPACITATION
        
    # MOTION_SICKNESS (High MAR or grimace + hand to mouth + head bobbing)
    if mar > 0.4 and hand_face_dist < 150.0 and abs(pitch) > 15.0:
        return MultimodalState.MOTION_SICKNESS
        
    # FEET_ON_DASHBOARD (Airbag hazard)
    if feet_dash:
        return MultimodalState.FEET_ON_DASHBOARD
        
    return MultimodalState.NOMINAL
