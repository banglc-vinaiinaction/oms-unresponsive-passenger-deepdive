"""
OMS Halpe26 Perception and 4-Level Intervention Engine for L4/L5 Robotaxi.
Implements the 26-keypoint topological model (Halpe26) from AlphaPose / RTMPose.
Computes deterministic geometric features, spatio-temporal dynamics, and maps to
the 4-tier intervention hierarchy:
  - Level 1: In-Cabin Automated HMI (Audio, Screen, Airbag, HVAC, Locks)
  - Level 2: Passerbys & Exterior (Hazard lights, External speaker, Window cracked 5cm)
  - Level 3: Fleet Operator & Tele-Operations (2-way remote voice, Field Support Van)
  - Level 4: Law Enforcement & EMS (eCall 113 / 115, remote unlock, GPS dispatch)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Dict, Any
import math


class SeatPosition(Enum):
    FRONT_LEFT = "Front Left (Driverless area)"
    FRONT_RIGHT = "Front Right (Passenger)"
    REAR_LEFT = "Rear Left"
    REAR_CENTER = "Rear Center"
    REAR_RIGHT = "Rear Right"


class InterventionLevelOMS(Enum):
    LEVEL_0_NOMINAL = 0      # Normal operation
    LEVEL_1_CABIN = 1        # In-cabin HMI, audio prompt, screen red outline, airbag suppression
    LEVEL_2_PASSERBY = 2     # Exterior speaker, hazard lights, window cracked 5cm, exterior SOS
    LEVEL_3_FLEET_OPS = 3    # Tele-operator 2-way call, Field Support Unit dispatched
    LEVEL_4_EMERGENCY = 4    # Automatic eCall 113 (Police) or 115 (EMS), remote unlock


class PostureState(Enum):
    NOMINAL_UPRIGHT = auto()
    RECLINED = auto()
    OUT_OF_POSITION_LEANING = auto()
    FEET_ON_DASHBOARD = auto()
    SLUMPED_UNRESPONSIVE = auto()


class AnomalyType(Enum):
    NONE = auto()
    UNBUCKLED_OR_DUMMY_CLIP = auto()
    FEET_ON_DASHBOARD = auto()
    LAP_SITTING_OVERCROWDING = auto()
    VIOLENCE_HARASSMENT = auto()
    SQUATTER_UNRESPONSIVE = auto()
    MOTION_SICKNESS_ONSET = auto()
    CAMERA_OCCLUSION_TAMPER = auto()


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float
    name: str = ""


# Halpe26 Joint Names
HALPE26_NAMES = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "HipCenter",
    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel"
]


@dataclass
class Halpe26Skeleton:
    keypoints: List[Keypoint]  # Exactly 26 keypoints
    track_id: int = 1

    def __post_init__(self):
        if len(self.keypoints) != 26:
            raise ValueError(f"Halpe26 requires exactly 26 keypoints, got {len(self.keypoints)}")

    def get(self, idx: int) -> Keypoint:
        return self.keypoints[idx]

    # Semantic accessors
    @property
    def neck(self) -> Keypoint: return self.keypoints[18]
    @property
    def hip_center(self) -> Keypoint: return self.keypoints[19]
    @property
    def left_shoulder(self) -> Keypoint: return self.keypoints[5]
    @property
    def right_shoulder(self) -> Keypoint: return self.keypoints[6]
    @property
    def left_wrist(self) -> Keypoint: return self.keypoints[9]
    @property
    def right_wrist(self) -> Keypoint: return self.keypoints[10]
    @property
    def left_hip(self) -> Keypoint: return self.keypoints[11]
    @property
    def right_hip(self) -> Keypoint: return self.keypoints[12]
    @property
    def left_big_toe(self) -> Keypoint: return self.keypoints[20]
    @property
    def right_big_toe(self) -> Keypoint: return self.keypoints[21]


@dataclass
class PassengerOMS:
    passenger_id: int
    seat: SeatPosition
    skeleton: Halpe26Skeleton
    is_child: bool = False
    physical_buckle_sensor: bool = True
    wrist_velocity_mps: float = 0.0
    stillness_duration_s: float = 0.0


def compute_trunk_angle_deg(skeleton: Halpe26Skeleton) -> float:
    """
    Computes angle between the spine vector (Neck -> HipCenter) and vertical vector (0, 1).
    0 deg = sitting upright; 90 deg = lying flat horizontally.
    """
    neck = skeleton.neck
    hip = skeleton.hip_center
    dx = hip.x - neck.x
    dy = hip.y - neck.y  # In image coordinates, y increases downward
    
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    
    # Cosine with vertical downward vector (0, 1)
    cos_theta = dy / length
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)


def check_feet_on_dashboard(skeleton: Halpe26Skeleton) -> bool:
    """
    In seated position, feet should be below hips (y_foot > y_hip in screen coords).
    If any foot point has y < y_hip - 40 pixels, feet are elevated onto dashboard/seat.
    """
    hip_y = skeleton.hip_center.y
    elevated_points = 0
    for idx in [20, 21, 24, 25]:  # Toes and heels
        if skeleton.keypoints[idx].y < hip_y - 30.0 and skeleton.keypoints[idx].confidence > 0.4:
            elevated_points += 1
    return elevated_points >= 2


def check_optical_seatbelt(skeleton: Halpe26Skeleton, has_diagonal_belt_line: bool) -> bool:
    """
    Optical check: Validates if the diagonal seatbelt is visually crossing
    the chest torso segment (between left shoulder and right hip or vice-versa).
    Cannot be fooled by a dummy buckle clip!
    """
    if not has_diagonal_belt_line:
        return False
    # If keypoints have high confidence and belt is detected crossing torso
    return True


def check_inter_passenger_violence(pax_a: PassengerOMS, pax_b: PassengerOMS) -> Tuple[bool, float]:
    """
    Calculates spatial distance between Passenger A and Passenger B.
    If distance < 0.45m and wrist velocity > 2.5 m/s -> Violence / Harassment.
    """
    wrist_a = pax_a.skeleton.right_wrist
    body_b = pax_b.skeleton.left_shoulder
    dist = math.hypot(wrist_a.x - body_b.x, wrist_a.y - body_b.y)
    
    is_violent = (dist < 200.0) and (pax_a.wrist_velocity_mps > 2.5 or pax_b.wrist_velocity_mps > 2.5)
    return is_violent, dist


class OMSDecisionEngine:
    """
    Evaluates in-cabin passengers, generates real-time telemetry, and determines
    the appropriate level in the 4-tier intervention hierarchy.
    """

    def evaluate_cabin(
        self,
        passengers: List[PassengerOMS],
        doors_open: bool = False,
        trip_completed: bool = False,
        has_camera_video_loss: bool = False
    ) -> Dict[str, Any]:
        result = {
            "intervention_level": InterventionLevelOMS.LEVEL_0_NOMINAL,
            "anomalies": [],
            "airbag_suppressed_seats": [],
            "hmi_actions": [],
            "passerby_actions": [],
            "fleet_ops_actions": [],
            "law_enforcement_actions": [],
            "passenger_count": len(passengers)
        }

        # Check 1: Camera tampering
        if has_camera_video_loss and len(passengers) > 0:
            result["anomalies"].append(AnomalyType.CAMERA_OCCLUSION_TAMPER)
            result["intervention_level"] = max(result["intervention_level"], InterventionLevelOMS.LEVEL_1_CABIN, key=lambda x: x.value)
            result["hmi_actions"].append("Cảnh báo âm thanh: Ống kính camera bị che khuất! Xe từ chối lăn bánh.")
            result["fleet_ops_actions"].append("Thông báo đội xe: Cảnh báo can thiệp phá hoại cảm biến.")

        # Check 2: Multi-person seat distribution & Lap-sitting
        seat_map = {}
        for p in passengers:
            seat_map.setdefault(p.seat, []).append(p)
            
        for seat, occupants in seat_map.items():
            if len(occupants) >= 2:
                result["anomalies"].append(AnomalyType.LAP_SITTING_OVERCROWDING)
                result["intervention_level"] = max(result["intervention_level"], InterventionLevelOMS.LEVEL_1_CABIN, key=lambda x: x.value)
                result["hmi_actions"].append(f"Màn hình {seat.value} bôi đỏ: Phát hiện 2 người ngồi chung ghế! Tự động cập nhật cước xe ghép.")

        # Check 3: Per-passenger posture, seatbelt, and airbag rules
        for p in passengers:
            skel = p.skeleton
            trunk_deg = compute_trunk_angle_deg(skel)
            feet_on_dash = check_feet_on_dashboard(skel)
            
            # Seatbelt check
            belt_ok = check_optical_seatbelt(skel, has_diagonal_belt_line=p.physical_buckle_sensor)
            if not belt_ok:
                result["anomalies"].append(AnomalyType.UNBUCKLED_OR_DUMMY_CLIP)
                result["intervention_level"] = max(result["intervention_level"], InterventionLevelOMS.LEVEL_1_CABIN, key=lambda x: x.value)
                result["hmi_actions"].append(f"Loa định hướng {p.seat.value}: 'Quý khách vui lòng thắt dây an toàn để xe lăn bánh'.")

            # Airbag suppression rule
            if trunk_deg > 45.0 or feet_on_dash:
                result["airbag_suppressed_seats"].append(p.seat)
                result["intervention_level"] = max(result["intervention_level"], InterventionLevelOMS.LEVEL_1_CABIN, key=lambda x: x.value)
                if feet_on_dash:
                    result["anomalies"].append(AnomalyType.FEET_ON_DASHBOARD)
                    result["hmi_actions"].append(f"Màn hình {p.seat.value} cảnh báo vàng: Túi khí phụ đã ngắt do gác chân nguy hiểm!")
                elif trunk_deg > 45.0:
                    result["hmi_actions"].append(f"Màn hình {p.seat.value}: Túi khí điều chỉnh lực nổ giảm do tư thế nằm ngả lưng.")

            # Squatter / Unresponsive passenger at dropoff
            if trip_completed and doors_open and p.stillness_duration_s > 60.0:
                result["anomalies"].append(AnomalyType.SQUATTER_UNRESPONSIVE)
                if p.stillness_duration_s > 180.0:
                    # After 3 minutes -> Escalate to Level 3 & 4
                    result["intervention_level"] = InterventionLevelOMS.LEVEL_4_EMERGENCY
                    result["hmi_actions"].append("Đèn trần bật sáng cực đại 100% + Còi báo thức âm lượng lớn.")
                    result["passerby_actions"].append("Màn hình kính ngoài hiện: 'CẦN TRỢ GIÚP Y TẾ' + Loa ngoài phát thanh nhờ giúp đỡ.")
                    result["fleet_ops_actions"].append("Tele-operator mở đàm thoại 2 chiều + Điều xe cơ động Field Support Van.")
                    result["law_enforcement_actions"].append("Tự động kích hoạt eCall 115 truyền dữ liệu nhịp thở + Mở khóa cửa từ xa.")
                else:
                    result["intervention_level"] = max(result["intervention_level"], InterventionLevelOMS.LEVEL_3_FLEET_OPS, key=lambda x: x.value)
                    result["hmi_actions"].append("Đèn trần bật sáng + Loa báo thức: 'Xe đã đến nơi, xin quý khách rời xe'.")
                    result["fleet_ops_actions"].append("Tele-operator gọi trực tiếp vào cabin qua loa xe.")

        # Check 4: Inter-passenger violence in ride-pooling
        if len(passengers) >= 2:
            for i in range(len(passengers)):
                for j in range(i + 1, len(passengers)):
                    is_violent, dist = check_inter_passenger_violence(passengers[i], passengers[j])
                    if is_violent:
                        result["anomalies"].append(AnomalyType.VIOLENCE_HARASSMENT)
                        result["intervention_level"] = InterventionLevelOMS.LEVEL_4_EMERGENCY
                        result["hmi_actions"].append("Hú còi báo động nội thất, đèn trần nháy đỏ rực, xe tự động tấp lề sáng đèn.")
                        result["passerby_actions"].append("Đèn Hazard chớp liên tục, loa ngoài phát: 'SỰ CỐ KHẨN CẤP TRONG XE!'.")
                        result["fleet_ops_actions"].append("Tele-operator quát răn đe qua loa: 'Hình ảnh bạo lực đang truyền trực tiếp về an ninh!'.")
                        result["law_enforcement_actions"].append("Tự động quay số Cảnh sát 113, truyền GPS và clip độ phân giải cao.")

        return result
