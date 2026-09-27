"""
Consolidated Final System Test Suite for Robotaxi OMS & DMS Intervention.
Validates:
  1. Full Multimodal Fusion (Nominal, Medical Emergency, Motion Sickness, Feet on Dash, Violence)
  2. Unimodal Fallback Behaviors (Face-occluded pose-only, Skeleton-occluded face-only, Dual sensor loss)
  3. Vehicle-Level Safe-State Fallbacks (Camera video loss stopped vs moving, Tele-op disconnection failsafe)
  4. CAN Bus Protocol E2E Compliance (ID 0x0F0 DIP_CMD & ID 0x130 OMS_STS with CRC8 SAE J1850 checks)
  5. Spatio-Temporal Progression Accuracy (0s, 60s, 120s, 150s, 180s)
"""

import unittest
import math
from typing import List, Tuple, Optional, Dict, Any

from src.face_pose_fusion import (
    FusedPassenger, MultimodalState, compute_hand_to_face_dist,
    evaluate_multimodal_passenger
)
from src.landmarks import FaceLandmarks, NUM_LANDMARKS
from src.oms_halpe import (
    Keypoint, Halpe26Skeleton, PassengerOMS, SeatPosition,
    OMSDecisionEngine, InterventionLevelOMS, AnomalyType,
    compute_trunk_angle_deg, check_feet_on_dashboard, check_inter_passenger_violence,
    HALPE26_NAMES
)
from src.can_frames import (
    CAN_ID_DIP_INTERVENTION, CAN_ID_OMS_STATUS, CAN_ID_DMS_STATUS,
    pack_dip_intervention_command, unpack_dip_intervention_command,
    pack_oms_status, unpack_oms_status,
    pack_dms_status, unpack_dms_status
)
from src.can_e2e import calculate_crc8_sae_j1850
from src.can_bus_interface import CANBusInterface, get_can_physical_voltages
from src.states import (
    InterventionLevel, AttentionState, GazeZone,
    DrowsinessLevel, DriverPresence, BeltStatus
)
from src.dms_features import compute_ear, compute_mar, estimate_head_pose


# ============================================================================
# Synthetic Landmark & Skeleton Generators
# ============================================================================

def make_mock_face(
    ear: float = 0.33,
    mar: float = 0.05,
    pitch_deg: float = 0.0,
    center_x: float = 100.0,
    center_y: float = 100.0
) -> FaceLandmarks:
    """
    Constructs a 50-point FaceLandmarks object with exact geometric EAR, MAR, and pitch.
    """
    pts = [(center_x, center_y)] * NUM_LANDMARKS

    # Eyebrows (0..4, 5..9)
    for i in range(5):
        pts[i] = (center_x - 20.0 + i * 4.0, center_y - 30.0)
        pts[5 + i] = (center_x + 5.0 + i * 4.0, center_y - 30.0)

    # Eyes at y = center_y - 15.0 (horizontal width = 20.0, vertical = ear * 20.0)
    eye_y = center_y - 15.0
    lx0, lx4 = center_x - 30.0, center_x - 10.0
    half_v = (ear * 20.0) / 2.0
    pts[14] = (lx0, eye_y)
    pts[18] = (lx4, eye_y)
    pts[15] = (lx0 + 5.0, eye_y - half_v)
    pts[16] = (lx0 + 10.0, eye_y - half_v)
    pts[17] = (lx0 + 15.0, eye_y - half_v)
    pts[21] = (lx0 + 5.0, eye_y + half_v)
    pts[20] = (lx0 + 10.0, eye_y + half_v)
    pts[19] = (lx0 + 15.0, eye_y + half_v)

    rx0, rx4 = center_x + 10.0, center_x + 30.0
    pts[22] = (rx0, eye_y)
    pts[26] = (rx4, eye_y)
    pts[23] = (rx0 + 5.0, eye_y - half_v)
    pts[24] = (rx0 + 10.0, eye_y - half_v)
    pts[25] = (rx0 + 15.0, eye_y - half_v)
    pts[29] = (rx0 + 5.0, eye_y + half_v)
    pts[28] = (rx0 + 10.0, eye_y + half_v)
    pts[27] = (rx0 + 15.0, eye_y + half_v)

    # Outer lips (30..41), horizontal width = 30.0, vertical = mar * 30.0
    mouth_y = center_y + 25.0
    mx0, mx6 = center_x - 15.0, center_x + 15.0
    half_m = (mar * 30.0) / 2.0
    pts[30] = (mx0, mouth_y)
    pts[36] = (mx6, mouth_y)
    pts[33] = (center_x, mouth_y - half_m)
    pts[39] = (center_x, mouth_y + half_m)
    pts[32] = (center_x - 7.5, mouth_y - half_m)
    pts[34] = (center_x + 7.5, mouth_y - half_m)
    pts[40] = (center_x - 7.5, mouth_y + half_m)
    pts[38] = (center_x + 7.5, mouth_y + half_m)
    pts[31] = (mx0 + 3.0, mouth_y - half_m * 0.5)
    pts[35] = (mx6 - 3.0, mouth_y - half_m * 0.5)
    pts[41] = (mx0 + 3.0, mouth_y + half_m * 0.5)
    pts[37] = (mx6 - 3.0, mouth_y + half_m * 0.5)

    # Inner lips (42..49)
    for i in range(8):
        pts[42 + i] = (center_x, mouth_y)

    # Nose (10..13): nose relative to face height (40px)
    # BASELINE = 0.65 -> nose_relative = (pitch_deg / -130.0) + 0.65
    nose_relative = (pitch_deg / -130.0) + 0.65
    nose_tip_y = eye_y + nose_relative * 40.0
    pts[10] = (center_x, eye_y + 10.0)
    pts[11] = (center_x, eye_y + 15.0)
    pts[12] = (center_x, eye_y + 20.0)
    pts[13] = (center_x, nose_tip_y)

    return FaceLandmarks(points=pts)


def make_mock_skeleton(
    trunk_angle_deg: float = 0.0,
    feet_elevated: bool = False,
    wrist_l: Tuple[float, float] = (70.0, 250.0),
    wrist_r: Tuple[float, float] = (130.0, 250.0),
    center_x: float = 100.0,
    center_y: float = 100.0
) -> Halpe26Skeleton:
    """
    Constructs a 26-keypoint Halpe26Skeleton with precise trunk angle and feet elevation.
    """
    kpts = [Keypoint(center_x, center_y, 0.9, name=n) for n in HALPE26_NAMES]
    neck_x = center_x
    neck_y = center_y - 48.0
    kpts[18] = Keypoint(neck_x, neck_y, 0.95, "Neck")
    kpts[17] = Keypoint(neck_x, neck_y - 25.0, 0.95, "Head")
    kpts[0] = Keypoint(neck_x, neck_y - 12.0, 0.95, "Nose")
    kpts[5] = Keypoint(center_x - 25.0, neck_y + 8.0, 0.92, "LShoulder")
    kpts[6] = Keypoint(center_x + 25.0, neck_y + 8.0, 0.92, "RShoulder")

    # Spine vector from neck to hip
    spine_len = 80.0
    rad = math.radians(trunk_angle_deg)
    hip_x = neck_x + spine_len * math.sin(rad)
    hip_y = neck_y + spine_len * math.cos(rad)
    kpts[19] = Keypoint(hip_x, hip_y, 0.92, "HipCenter")
    kpts[11] = Keypoint(hip_x - 15.0, hip_y, 0.90, "LHip")
    kpts[12] = Keypoint(hip_x + 15.0, hip_y, 0.90, "RHip")
    kpts[9] = Keypoint(wrist_l[0], wrist_l[1], 0.9, "LWrist")
    kpts[10] = Keypoint(wrist_r[0], wrist_r[1], 0.9, "RWrist")

    # Knees
    kpts[13] = Keypoint(hip_x - 15.0, hip_y + 40.0, 0.88, "LKnee")
    kpts[14] = Keypoint(hip_x + 15.0, hip_y + 40.0, 0.88, "RKnee")

    # Feet
    foot_y = hip_y - 45.0 if feet_elevated else hip_y + 80.0
    for idx, name in [(20, "LBigToe"), (21, "RBigToe"), (22, "LSmallToe"),
                      (23, "RSmallToe"), (24, "LHeel"), (25, "RHeel")]:
        offset_x = -15.0 if "L" in name else 15.0
        kpts[idx] = Keypoint(hip_x + offset_x, foot_y, 0.9, name)

    return Halpe26Skeleton(keypoints=kpts)


# ============================================================================
# 1. Full Multimodal Fusion Suite
# ============================================================================

class TestMultimodalFusionSuite(unittest.TestCase):
    """
    Verifies full multimodal fusion across 5 core states and priority ordering:
      - NOMINAL
      - MEDICAL_INCAPACITATION
      - MOTION_SICKNESS
      - FEET_ON_DASHBOARD
      - HARASSMENT_VIOLENCE
    """

    def test_multimodal_nominal_passenger(self):
        """Alert eyes (EAR~0.33), closed mouth (MAR~0.05), upright torso (trunk 0°) -> NOMINAL."""
        face = make_mock_face(ear=0.33, mar=0.05, pitch_deg=0.0)
        skel = make_mock_skeleton(trunk_angle_deg=0.0, feet_elevated=False)
        pax = FusedPassenger(passenger_id=1, landmarks=face, skeleton=skel)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.NOMINAL)

    def test_multimodal_medical_emergency_collapse(self):
        """Eyes drooping/closed (EAR < 0.20) + trunk collapse (trunk > 40°) -> MEDICAL_INCAPACITATION."""
        face = make_mock_face(ear=0.08, mar=0.05, pitch_deg=-10.0)
        skel = make_mock_skeleton(trunk_angle_deg=50.0, feet_elevated=False)
        pax = FusedPassenger(passenger_id=1, landmarks=face, skeleton=skel)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)

    def test_multimodal_motion_sickness_onset(self):
        """Mouth wide open/grimacing (MAR > 0.40) + hand near mouth (<150px) + head pitch (>15°) -> MOTION_SICKNESS."""
        face = make_mock_face(ear=0.30, mar=0.55, pitch_deg=-20.0)
        # Position left wrist at mouth coordinates (100.0, 125.0)
        skel = make_mock_skeleton(trunk_angle_deg=10.0, feet_elevated=False, wrist_l=(100.0, 125.0))
        pax = FusedPassenger(passenger_id=1, landmarks=face, skeleton=skel)

        dist = compute_hand_to_face_dist(pax.skeleton, pax.landmarks)
        self.assertLess(dist, 150.0)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MOTION_SICKNESS)

    def test_multimodal_feet_on_dashboard_airbag_hazard(self):
        """Alert passenger with feet propped onto dashboard (y_foot < y_hip - 30) -> FEET_ON_DASHBOARD."""
        face = make_mock_face(ear=0.33, mar=0.05, pitch_deg=0.0)
        skel = make_mock_skeleton(trunk_angle_deg=5.0, feet_elevated=True)
        pax = FusedPassenger(passenger_id=1, landmarks=face, skeleton=skel)

        self.assertTrue(check_feet_on_dashboard(pax.skeleton))
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.FEET_ON_DASHBOARD)

    def test_multimodal_inter_passenger_violence(self):
        """Two passengers with wrist distance < 350px and velocity > 2.5 m/s -> HARASSMENT_VIOLENCE."""
        skel_a = make_mock_skeleton(wrist_r=(150.0, 150.0))
        skel_b = make_mock_skeleton(wrist_l=(170.0, 150.0))
        pax_a = FusedPassenger(1, make_mock_face(), skel_a, wrist_velocity_mps=3.2)
        pax_b = FusedPassenger(2, make_mock_face(), skel_b, wrist_velocity_mps=3.0)

        state_a = evaluate_multimodal_passenger(pax_a, all_passengers=[pax_a, pax_b])
        state_b = evaluate_multimodal_passenger(pax_b, all_passengers=[pax_a, pax_b])

        self.assertEqual(state_a, MultimodalState.HARASSMENT_VIOLENCE)
        self.assertEqual(state_b, MultimodalState.HARASSMENT_VIOLENCE)

    def test_multimodal_priority_violence_over_medical_and_sickness(self):
        """Violence has top priority even if trunk collapse or sickness flags are present."""
        skel_a = make_mock_skeleton(trunk_angle_deg=55.0, wrist_r=(150.0, 150.0))
        skel_b = make_mock_skeleton(wrist_l=(170.0, 150.0))
        face_a = make_mock_face(ear=0.05, mar=0.60, pitch_deg=-25.0)

        pax_a = FusedPassenger(1, face_a, skel_a, wrist_velocity_mps=3.5)
        pax_b = FusedPassenger(2, make_mock_face(), skel_b, wrist_velocity_mps=2.8)

        state = evaluate_multimodal_passenger(pax_a, all_passengers=[pax_a, pax_b])
        self.assertEqual(state, MultimodalState.HARASSMENT_VIOLENCE)

    def test_multimodal_priority_motion_sickness_over_feet_dashboard(self):
        """Motion sickness condition is prioritized over feet on dashboard to prevent masking emesis."""
        face = make_mock_face(ear=0.30, mar=0.55, pitch_deg=-20.0)
        skel = make_mock_skeleton(trunk_angle_deg=10.0, feet_elevated=True, wrist_l=(100.0, 125.0))
        pax = FusedPassenger(1, face, skel)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MOTION_SICKNESS)


# ============================================================================
# 2. Unimodal Fallback Matrix Suite
# ============================================================================

class TestUnimodalFallbackSuite(unittest.TestCase):
    """
    Verifies graceful sensor degradation across:
      - Face-occluded (Pose-only fallback)
      - Skeleton-occluded (Face-only fallback)
      - Dual sensor loss (Complete degradation fallback)
    """

    def test_fallback_face_occluded_nominal(self):
        """Face camera occluded, normal upright body pose -> NOMINAL."""
        skel = make_mock_skeleton(trunk_angle_deg=0.0, feet_elevated=False)
        pax = FusedPassenger(1, landmarks=None, skeleton=skel, is_face_occluded=True)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.NOMINAL)

    def test_fallback_face_occluded_trunk_collapse_emergency(self):
        """Face camera occluded, trunk angle > 40° -> MEDICAL_INCAPACITATION."""
        skel = make_mock_skeleton(trunk_angle_deg=48.0, feet_elevated=False)
        pax = FusedPassenger(1, landmarks=None, skeleton=skel, is_face_occluded=True)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)

    def test_fallback_face_occluded_feet_dashboard(self):
        """Face camera occluded, feet elevated onto dashboard -> FEET_ON_DASHBOARD."""
        skel = make_mock_skeleton(trunk_angle_deg=10.0, feet_elevated=True)
        pax = FusedPassenger(1, landmarks=None, skeleton=skel, is_face_occluded=True)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.FEET_ON_DASHBOARD)

    def test_fallback_face_occluded_violence(self):
        """Face camera occluded, violence detected via skeleton tracking -> HARASSMENT_VIOLENCE."""
        skel_a = make_mock_skeleton(wrist_r=(150.0, 150.0))
        skel_b = make_mock_skeleton(wrist_l=(160.0, 150.0))
        pax_a = FusedPassenger(1, None, skel_a, wrist_velocity_mps=3.2, is_face_occluded=True)
        pax_b = FusedPassenger(2, None, skel_b, wrist_velocity_mps=3.0, is_face_occluded=True)

        state = evaluate_multimodal_passenger(pax_a, all_passengers=[pax_a, pax_b])
        self.assertEqual(state, MultimodalState.HARASSMENT_VIOLENCE)

    def test_fallback_skeleton_occluded_nominal(self):
        """Body occluded by blanket/luggage, alert upright face -> NOMINAL."""
        face = make_mock_face(ear=0.32, mar=0.05, pitch_deg=0.0)
        pax = FusedPassenger(1, landmarks=face, skeleton=None, is_skeleton_occluded=True)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.NOMINAL)

    def test_fallback_skeleton_occluded_pitch_drop_emergency(self):
        """Body occluded, drooping eyes (EAR < 0.20) + head dropped (pitch < -15°) -> MEDICAL_INCAPACITATION."""
        face = make_mock_face(ear=0.15, mar=0.05, pitch_deg=-20.0)
        pax = FusedPassenger(1, landmarks=face, skeleton=None, is_skeleton_occluded=True)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)

    def test_fallback_skeleton_occluded_eyes_shut_emergency(self):
        """Body occluded, eyes completely closed (EAR < 0.10) -> MEDICAL_INCAPACITATION regardless of pitch."""
        face = make_mock_face(ear=0.05, mar=0.05, pitch_deg=0.0)
        pax = FusedPassenger(1, landmarks=face, skeleton=None, is_skeleton_occluded=True)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)

    def test_fallback_dual_sensor_loss_both_none(self):
        """Both face and skeleton sensors lost/None -> SENSOR_OCCLUDED_FALLBACK."""
        pax = FusedPassenger(1, landmarks=None, skeleton=None)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.SENSOR_OCCLUDED_FALLBACK)

    def test_fallback_dual_sensor_loss_both_flags_occluded(self):
        """Both sensors present but flagged occluded -> SENSOR_OCCLUDED_FALLBACK."""
        face = make_mock_face()
        skel = make_mock_skeleton()
        pax = FusedPassenger(1, face, skel, is_face_occluded=True, is_skeleton_occluded=True)

        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.SENSOR_OCCLUDED_FALLBACK)

    def test_fallback_dual_sensor_insufficient_keypoints(self):
        """Degraded points (< 50 landmarks or < 20 keypoints) triggers fallback."""
        face_short = FaceLandmarks(points=[(0.0, 0.0)] * 50)
        face_short.points = face_short.points[:30]  # truncate to 30

        pax = FusedPassenger(1, landmarks=face_short, skeleton=None)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.SENSOR_OCCLUDED_FALLBACK)


# ============================================================================
# 3. Vehicle-Level Safe-State Fallbacks Suite
# ============================================================================

class TestVehicleSafeStateSuite(unittest.TestCase):
    """
    Verifies vehicle-level fail-safe actuation:
      - Camera video loss when vehicle is stopped -> Refuse departure, Level 3 Fleet Ops
      - Camera video loss when vehicle is moving -> MRM safe pull-over, Level 3 Fleet Ops, Hazard
      - Tele-op disconnected failsafe fallback -> Local autonomous safe-stop, mechanical unlock, SOS horn
    """

    def setUp(self):
        self.engine = OMSDecisionEngine()

    def test_camera_loss_stopped_refuse_departure(self):
        """Camera loss at stop/departure: refuses departure, LEVEL_1 cabin alert, LEVEL_3 tele-op escalation."""
        pax = PassengerOMS(1, SeatPosition.FRONT_RIGHT, make_mock_skeleton())
        res = self.engine.evaluate_cabin(
            passengers=[pax],
            doors_open=False,
            trip_completed=False,
            has_camera_video_loss=True,
            vehicle_in_motion=False
        )

        self.assertIn(AnomalyType.CAMERA_OCCLUSION_TAMPER, res["anomalies"])
        self.assertEqual(res["intervention_level"], InterventionLevelOMS.LEVEL_3_FLEET_OPS)
        self.assertTrue(res["refuse_departure"])
        self.assertFalse(res["mrm_takeover"])
        self.assertTrue(any("từ chối lăn bánh" in act or "Refuses to depart" in act for act in res["hmi_actions"]))
        self.assertTrue(any("Tele-operator" in act for act in res["fleet_ops_actions"]))

    def test_camera_loss_moving_mrm_pullover(self):
        """Camera loss in motion: executes MRM safe pull-over, LEVEL_1 cabin alert, LEVEL_3 tele-op, Hazard."""
        pax = PassengerOMS(1, SeatPosition.REAR_LEFT, make_mock_skeleton())
        res = self.engine.evaluate_cabin(
            passengers=[pax],
            doors_open=False,
            trip_completed=False,
            has_camera_video_loss=True,
            vehicle_in_motion=True
        )

        self.assertIn(AnomalyType.CAMERA_OCCLUSION_TAMPER, res["anomalies"])
        self.assertEqual(res["intervention_level"], InterventionLevelOMS.LEVEL_3_FLEET_OPS)
        self.assertTrue(res["mrm_takeover"])
        self.assertFalse(res["refuse_departure"])
        self.assertTrue(any("MRM" in act or "tấp lề an toàn" in act for act in res["hmi_actions"]))
        self.assertTrue(any("Hazard" in act for act in res["passerby_actions"]))
        self.assertTrue(any("Tele-operator" in act for act in res["fleet_ops_actions"]))

    def test_camera_loss_with_empty_passengers(self):
        """Camera loss with no passengers detected still flags tamper anomaly and refuses departure."""
        res = self.engine.evaluate_cabin(
            passengers=[],
            has_camera_video_loss=True,
            vehicle_in_motion=False
        )
        self.assertIn(AnomalyType.CAMERA_OCCLUSION_TAMPER, res["anomalies"])
        self.assertEqual(res["intervention_level"], InterventionLevelOMS.LEVEL_3_FLEET_OPS)
        self.assertTrue(res["refuse_departure"])

    def test_unresponsive_passenger_teleop_connected_escalation(self):
        """Unresponsive passenger after 180s with active tele-op escalates to Level 4 with 2-way call + eCall."""
        pax = PassengerOMS(1, SeatPosition.REAR_LEFT, make_mock_skeleton(trunk_angle_deg=50.0), stillness_duration_s=180.0)
        res = self.engine.evaluate_cabin(
            passengers=[pax],
            doors_open=True,
            trip_completed=True,
            teleop_link_lost=False
        )

        self.assertEqual(res["intervention_level"], InterventionLevelOMS.LEVEL_4_EMERGENCY)
        self.assertIn(AnomalyType.SQUATTER_UNRESPONSIVE, res["anomalies"])
        self.assertIsNone(res["failsafe_fallback"])
        self.assertTrue(any("Tele-operator" in act and "đàm thoại" in act for act in res["fleet_ops_actions"]))
        self.assertTrue(any("eCall 115" in act for act in res["law_enforcement_actions"]))

    def test_unresponsive_passenger_teleop_lost_autonomous_safe_stop(self):
        """Unresponsive passenger after 180s with tele-op lost triggers autonomous safe-stop & door unlock."""
        expected_fallback = (
            "Autonomous Safe-Stop & Door Unlock Fallback: "
            "Xe tự tấp lề an toàn, mở khóa tất cả cửa xe từ chốt cơ điện tử, "
            "kích hoạt còi báo động SOS và đèn hazard ngoài xe."
        )

        pax = PassengerOMS(1, SeatPosition.REAR_LEFT, make_mock_skeleton(trunk_angle_deg=50.0), stillness_duration_s=180.0)
        res = self.engine.evaluate_cabin(
            passengers=[pax],
            doors_open=True,
            trip_completed=True,
            teleop_link_lost=True
        )

        self.assertEqual(res["intervention_level"], InterventionLevelOMS.LEVEL_4_EMERGENCY)
        self.assertIn(AnomalyType.SQUATTER_UNRESPONSIVE, res["anomalies"])
        self.assertEqual(res["failsafe_fallback"], expected_fallback)
        self.assertIn(expected_fallback, res["law_enforcement_actions"])
        self.assertTrue(res["mrm_takeover"])
        self.assertTrue(any("Mất kết nối" in act for act in res["fleet_ops_actions"]))
        self.assertTrue(any("còi báo động SOS" in act for act in res["passerby_actions"]))

    def test_violence_teleop_lost_local_failsafe(self):
        """Violence with tele-op link lost executes local failsafe fallback and records video clip."""
        skel_a = make_mock_skeleton(wrist_r=(100.0, 100.0))
        skel_b = make_mock_skeleton(wrist_l=(105.0, 100.0))
        # Place shoulder of B close to wrist of A
        skel_b.keypoints[5] = Keypoint(105.0, 100.0, 0.9, "LShoulder")

        pax_a = PassengerOMS(1, SeatPosition.REAR_LEFT, skel_a, wrist_velocity_mps=3.2)
        pax_b = PassengerOMS(2, SeatPosition.REAR_RIGHT, skel_b, wrist_velocity_mps=3.0)

        res = self.engine.evaluate_cabin(
            passengers=[pax_a, pax_b],
            teleop_link_lost=True
        )

        self.assertIn(AnomalyType.VIOLENCE_HARASSMENT, res["anomalies"])
        self.assertEqual(res["intervention_level"], InterventionLevelOMS.LEVEL_4_EMERGENCY)
        self.assertIsNotNone(res["failsafe_fallback"])
        self.assertTrue(res["mrm_takeover"])
        self.assertTrue(any("ghi lại clip" in act for act in res["fleet_ops_actions"]))


# ============================================================================
# 4. CAN Bus Protocol E2E Compliance Suite
# ============================================================================

class TestCANBusProtocolSuite(unittest.TestCase):
    """
    Verifies CAN Bus protocol compliance:
      - ID 0x0F0 DIP_CMD packing, unpacking, field roundtrip
      - ID 0x130 OMS_STS packing, unpacking, field roundtrip
      - CRC8 SAE J1850 mathematical verification and tamper rejection
      - Alive counter rolling sequence
      - Virtual SocketCAN loopback interface
      - ISO 11898-2 physical line voltages
    """

    def test_dip_intervention_cmd_pack_unpack_roundtrip(self):
        """Validates ID 0x0F0 packing/unpacking bitfields and decel precision."""
        packed = pack_dip_intervention_command(
            intervention_level=InterventionLevel.LEVEL_5_MRM_STOP,
            visual_alert=2,
            acoustic_alert=3,
            haptic_wheel=True,
            seatbelt_tug=True,
            brake_jerk=False,
            hazard_flash=True,
            mrm_takeover=True,
            ecall_trigger=True,
            target_decel_mps2=2.25,
            alive_counter=9
        )
        self.assertEqual(len(packed), 8)

        unpacked = unpack_dip_intervention_command(packed)
        self.assertEqual(unpacked["intervention_level"], InterventionLevel.LEVEL_5_MRM_STOP)
        self.assertEqual(unpacked["visual_alert"], 2)
        self.assertEqual(unpacked["acoustic_alert"], 3)
        self.assertTrue(unpacked["haptic_wheel"])
        self.assertTrue(unpacked["seatbelt_tug"])
        self.assertFalse(unpacked["brake_jerk"])
        self.assertTrue(unpacked["hazard_flash"])
        self.assertTrue(unpacked["mrm_takeover"])
        self.assertTrue(unpacked["ecall_trigger"])
        self.assertAlmostEqual(unpacked["target_decel_mps2"], 2.25, places=2)
        self.assertEqual(unpacked["alive_counter"], 9)
        self.assertEqual(unpacked["crc8"], packed[7])

    def test_dip_intervention_cmd_crc8_tamper_rejection(self):
        """Tampering any bit in ID 0x0F0 payload triggers CRC failure."""
        valid_frame = pack_dip_intervention_command(
            intervention_level=InterventionLevel.LEVEL_3_HAPTIC,
            visual_alert=1,
            acoustic_alert=1,
            haptic_wheel=False,
            seatbelt_tug=True,
            brake_jerk=False,
            hazard_flash=False,
            mrm_takeover=False,
            ecall_trigger=False,
            target_decel_mps2=0.0,
            alive_counter=3
        )
        # Clean frame unpacks without error
        unpacked = unpack_dip_intervention_command(valid_frame)
        self.assertIsNotNone(unpacked)

        # Corrupt data byte 0
        corrupted_b0 = bytes([valid_frame[0] ^ 0x01]) + valid_frame[1:]
        with self.assertRaises(ValueError):
            unpack_dip_intervention_command(corrupted_b0)

        # Corrupt CRC byte 7 directly
        corrupted_crc = valid_frame[:7] + bytes([valid_frame[7] ^ 0xFF])
        with self.assertRaises(ValueError):
            unpack_dip_intervention_command(corrupted_crc)

    def test_oms_status_pack_unpack_roundtrip(self):
        """Validates ID 0x130 cabin telemetry packing/unpacking and CRC."""
        packed = pack_oms_status(
            driver_present=DriverPresence.PRESENT,
            driver_belt=BeltStatus.BUCKLED,
            passenger_count=3,
            child_present=True,
            alive_counter=14
        )
        self.assertEqual(len(packed), 8)

        unpacked = unpack_oms_status(packed)
        self.assertEqual(unpacked["driver_present"], DriverPresence.PRESENT)
        self.assertEqual(unpacked["driver_belt_fastened"], BeltStatus.BUCKLED)
        self.assertEqual(unpacked["passenger_count"], 3)
        self.assertTrue(unpacked["child_present"])
        self.assertEqual(unpacked["alive_counter"], 14)
        self.assertEqual(unpacked["crc8"], packed[7])

    def test_oms_status_crc8_tamper_rejection(self):
        """Tampering ID 0x130 passenger count or CRC raises ValueError."""
        valid_frame = pack_oms_status(
            driver_present=DriverPresence.ABSENT,
            driver_belt=BeltStatus.UNBUCKLED,
            passenger_count=1,
            child_present=False,
            alive_counter=5
        )
        # Tamper byte 0
        corrupted = bytes([valid_frame[0] ^ 0x04]) + valid_frame[1:]
        with self.assertRaises(ValueError):
            unpack_oms_status(corrupted)

    def test_can_frame_invalid_length_rejection(self):
        """Non-8-byte frames must be rejected immediately."""
        short_frame = b"\x01\x02\x03"
        long_frame = b"\x00" * 9
        with self.assertRaises(ValueError):
            unpack_dip_intervention_command(short_frame)
        with self.assertRaises(ValueError):
            unpack_dip_intervention_command(long_frame)
        with self.assertRaises(ValueError):
            unpack_oms_status(short_frame)
        with self.assertRaises(ValueError):
            unpack_oms_status(long_frame)

    def test_alive_counter_rolling_sequence_wrap(self):
        """Validates alive counter rolls monotonically 0..15 and wraps back to 0."""
        counters = []
        for i in range(32):
            frame = pack_dip_intervention_command(
                intervention_level=InterventionLevel.LEVEL_0_PASSIVE,
                visual_alert=0, acoustic_alert=0, haptic_wheel=False,
                seatbelt_tug=False, brake_jerk=False, hazard_flash=False,
                mrm_takeover=False, ecall_trigger=False, target_decel_mps2=0.0,
                alive_counter=i
            )
            data = unpack_dip_intervention_command(frame)
            counters.append(data["alive_counter"])

        expected = [i % 16 for i in range(32)]
        self.assertEqual(counters, expected)

    def test_virtual_can_bus_loopback_e2e(self):
        """Validates E2E frame transmission and reception across CANBusInterface."""
        bus = CANBusInterface(channel="vcan_test0", virtual_fallback=True)
        self.assertTrue(bus.is_virtual)

        # Transmit 0x0F0
        cmd_bytes = pack_dip_intervention_command(
            intervention_level=InterventionLevel.LEVEL_4_BRAKE_JERK,
            visual_alert=2, acoustic_alert=2, haptic_wheel=False,
            seatbelt_tug=False, brake_jerk=True, hazard_flash=True,
            mrm_takeover=False, ecall_trigger=False, target_decel_mps2=1.0,
            alive_counter=2
        )
        bus.send(arbitration_id=CAN_ID_DIP_INTERVENTION, payload=cmd_bytes)

        # Transmit 0x130
        oms_bytes = pack_oms_status(
            driver_present=DriverPresence.PRESENT,
            driver_belt=BeltStatus.BUCKLED,
            passenger_count=2,
            child_present=False,
            alive_counter=2
        )
        bus.send(arbitration_id=CAN_ID_OMS_STATUS, payload=oms_bytes)

        # Receive 0x0F0
        msg1 = bus.recv()
        self.assertIsNotNone(msg1)
        self.assertEqual(msg1.arbitration_id, CAN_ID_DIP_INTERVENTION)
        unpacked1 = unpack_dip_intervention_command(msg1.data)
        self.assertEqual(unpacked1["intervention_level"], InterventionLevel.LEVEL_4_BRAKE_JERK)

        # Receive 0x130
        msg2 = bus.recv()
        self.assertIsNotNone(msg2)
        self.assertEqual(msg2.arbitration_id, CAN_ID_OMS_STATUS)
        unpacked2 = unpack_oms_status(msg2.data)
        self.assertEqual(unpacked2["passenger_count"], 2)

        # Queue empty
        self.assertIsNone(bus.recv())
        bus.close()

    def test_can_physical_layer_voltages(self):
        """Verifies ISO 11898-2 physical line voltages for Dominant (3.5/1.5V) and Recessive (2.5/2.5V)."""
        byte_seq = bytes([0xAA])  # 10101010 in binary
        voltages = get_can_physical_voltages(byte_seq)
        self.assertEqual(len(voltages), 8)

        # Bit 0 (MSB of 0xAA) is '1' -> Recessive
        self.assertEqual(voltages[0].bit_value, 1)
        self.assertEqual(voltages[0].bus_state, "RECESSIVE")
        self.assertEqual(voltages[0].v_can_h, 2.5)
        self.assertEqual(voltages[0].v_can_l, 2.5)
        self.assertEqual(voltages[0].v_diff, 0.0)

        # Bit 1 is '0' -> Dominant
        self.assertEqual(voltages[1].bit_value, 0)
        self.assertEqual(voltages[1].bus_state, "DOMINANT")
        self.assertEqual(voltages[1].v_can_h, 3.5)
        self.assertEqual(voltages[1].v_can_l, 1.5)
        self.assertEqual(voltages[1].v_diff, 2.0)


# ============================================================================
# 5. Spatio-Temporal Progression Accuracy Suite
# ============================================================================

class TestTemporalProgressionSuite(unittest.TestCase):
    """
    Verifies temporal escalation timeline:
      - 0s: LEVEL_0_NOMINAL (Normal arrival)
      - 60s: LEVEL_1_CABIN (In-cabin voice and screen prompt)
      - 120s: LEVEL_2_PASSERBY (Exterior glass SOS display + speaker)
      - 150s: LEVEL_3_FLEET_OPS (Tele-operator live cabin intercom)
      - 180s: LEVEL_4_EMERGENCY (Full strobe/horn, eCall 115, door unlock)
      - Monotonic escalation property
      - Recovery de-escalation
    """

    def setUp(self):
        self.engine = OMSDecisionEngine()

    def test_temporal_timeline_progression_accuracy(self):
        """Validates exact intervention level and actuation at 0s, 60s, 120s, 150s, 180s."""
        passenger = PassengerOMS(
            passenger_id=1,
            seat=SeatPosition.REAR_LEFT,
            skeleton=make_mock_skeleton(trunk_angle_deg=0.0),
            physical_buckle_sensor=True,
            stillness_duration_s=0.0
        )

        timeline = [
            (0.0, InterventionLevelOMS.LEVEL_0_NOMINAL, False, False, False, False),
            (60.0, InterventionLevelOMS.LEVEL_1_CABIN, True, False, False, False),
            (120.0, InterventionLevelOMS.LEVEL_2_PASSERBY, True, True, False, False),
            (150.0, InterventionLevelOMS.LEVEL_3_FLEET_OPS, True, False, True, False),
            (180.0, InterventionLevelOMS.LEVEL_4_EMERGENCY, True, True, True, True),
        ]

        for (t, exp_level, exp_hmi, exp_passerby, exp_fleet, exp_emergency) in timeline:
            passenger.stillness_duration_s = t
            res = self.engine.evaluate_cabin(
                passengers=[passenger],
                doors_open=True,
                trip_completed=True,
                has_camera_video_loss=False,
                teleop_link_lost=False
            )

            # Verification of level
            self.assertEqual(
                res["intervention_level"], exp_level,
                f"Mismatch at t={t}s: expected {exp_level}, got {res['intervention_level']}"
            )

            # Verification of actuation activation
            if exp_hmi:
                self.assertTrue(len(res["hmi_actions"]) > 0, f"HMI missing at t={t}")
            if exp_passerby:
                self.assertTrue(len(res["passerby_actions"]) > 0, f"Passerby action missing at t={t}")
            if exp_fleet:
                self.assertTrue(len(res["fleet_ops_actions"]) > 0, f"Fleet ops missing at t={t}")
            if exp_emergency:
                self.assertTrue(len(res["law_enforcement_actions"]) > 0, f"LE/eCall missing at t={t}")

    def test_temporal_escalation_strictly_monotonic(self):
        """Verifies intervention level is monotonically non-decreasing over time."""
        passenger = PassengerOMS(
            passenger_id=1,
            seat=SeatPosition.REAR_LEFT,
            skeleton=make_mock_skeleton(trunk_angle_deg=50.0),
            physical_buckle_sensor=True,
            stillness_duration_s=0.0
        )

        sample_times = [0, 15, 30, 60, 90, 120, 140, 150, 160, 180, 240]
        previous_level = 0

        for t in sample_times:
            passenger.stillness_duration_s = float(t)
            res = self.engine.evaluate_cabin(
                passengers=[passenger],
                doors_open=True,
                trip_completed=True
            )
            current_level = res["intervention_level"].value
            self.assertGreaterEqual(
                current_level, previous_level,
                f"Escalation non-monotonic at t={t}s: previous={previous_level}, current={current_level}"
            )
            previous_level = current_level

    def test_passenger_recovery_deescalates(self):
        """When an unresponsive passenger moves (stillness resets), system de-escalates to nominal."""
        passenger = PassengerOMS(
            passenger_id=1,
            seat=SeatPosition.REAR_RIGHT,
            skeleton=make_mock_skeleton(trunk_angle_deg=0.0),
            physical_buckle_sensor=True,
            stillness_duration_s=180.0
        )
        # Peak emergency
        res_crit = self.engine.evaluate_cabin(
            passengers=[passenger],
            doors_open=True,
            trip_completed=True
        )
        self.assertEqual(res_crit["intervention_level"], InterventionLevelOMS.LEVEL_4_EMERGENCY)

        # Passenger wakes up and moves
        passenger.stillness_duration_s = 0.0
        res_recovered = self.engine.evaluate_cabin(
            passengers=[passenger],
            doors_open=True,
            trip_completed=True
        )
        self.assertEqual(res_recovered["intervention_level"], InterventionLevelOMS.LEVEL_0_NOMINAL)
        self.assertNotIn(AnomalyType.SQUATTER_UNRESPONSIVE, res_recovered["anomalies"])


if __name__ == '__main__':
    unittest.main()
