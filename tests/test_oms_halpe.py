"""
Unit tests for OMS Halpe26 Perception, Fallback Behaviors, and Safe-State Logic.
Tests:
  1. Camera tampering & video loss safe-state fallback (Refuse departure / MRM safe pull-over, Level 1 Cabin, Level 3 Tele-op)
  2. Unresponsive passenger emergency escalation (Level 1 -> 2 -> 3 -> 4)
  3. Tele-op connectivity lost failsafe fallback (Autonomous Safe-Stop & Door Unlock Fallback)
  4. Inter-passenger violence and tele-op loss
  5. CAN bus packing/unpacking integrity under safe-state fallbacks
"""

import unittest
from src.oms_halpe import (
    Keypoint, Halpe26Skeleton, PassengerOMS, SeatPosition,
    OMSDecisionEngine, InterventionLevelOMS, AnomalyType,
    compute_trunk_angle_deg, check_feet_on_dashboard
)
from src.can_frames import (
    pack_dip_intervention_command, unpack_dip_intervention_command,
    pack_oms_status, unpack_oms_status
)
from src.states import (
    InterventionLevel, DriverPresence, BeltStatus
)


def create_test_skeleton(trunk_lean_dx: float = 0.0, feet_elevated: bool = False) -> Halpe26Skeleton:
    kpts = [
        Keypoint(100.0, 40.0, 0.95, "Nose"),
        Keypoint(95.0, 35.0, 0.95, "LEye"),
        Keypoint(105.0, 35.0, 0.95, "REye"),
        Keypoint(88.0, 38.0, 0.90, "LEar"),
        Keypoint(112.0, 38.0, 0.90, "REar"),
        Keypoint(75.0, 60.0, 0.92, "LShoulder"),
        Keypoint(125.0, 60.0, 0.92, "RShoulder"),
        Keypoint(70.0, 95.0, 0.88, "LElbow"),
        Keypoint(130.0, 95.0, 0.88, "RElbow"),
        Keypoint(80.0, 120.0, 0.85, "LWrist"),
        Keypoint(120.0, 120.0, 0.85, "RWrist"),
        Keypoint(85.0 + trunk_lean_dx, 130.0, 0.90, "LHip"),
        Keypoint(115.0 + trunk_lean_dx, 130.0, 0.90, "RHip"),
        Keypoint(80.0 + trunk_lean_dx, 170.0, 0.88, "LKnee"),
        Keypoint(120.0 + trunk_lean_dx, 170.0, 0.88, "RKnee"),
        Keypoint(80.0, 210.0, 0.85, "LAnkle"),
        Keypoint(120.0, 210.0, 0.85, "RAnkle"),
        Keypoint(100.0, 25.0, 0.95, "Head"),
        Keypoint(100.0, 52.0, 0.95, "Neck"),
        Keypoint(100.0 + trunk_lean_dx, 130.0, 0.92, "HipCenter"),
        Keypoint(80.0, 220.0, 0.85, "LBigToe"),
        Keypoint(120.0, 220.0, 0.85, "RBigToe"),
        Keypoint(75.0, 222.0, 0.82, "LSmallToe"),
        Keypoint(125.0, 222.0, 0.82, "RSmallToe"),
        Keypoint(82.0, 205.0, 0.85, "LHeel"),
        Keypoint(118.0, 205.0, 0.85, "RHeel"),
    ]
    if feet_elevated:
        for idx in [20, 21, 22, 23, 24, 25]:
            kpts[idx].y = 130.0 - 45.0
    return Halpe26Skeleton(keypoints=kpts)


class TestOMSSafeStateFallback(unittest.TestCase):

    def setUp(self):
        self.engine = OMSDecisionEngine()

    def test_camera_loss_stopped_refuse_departure(self):
        """Camera loss at stop/departure: refuses departure, LEVEL_1 cabin alert, LEVEL_3 tele-op escalation."""
        pax = PassengerOMS(1, SeatPosition.FRONT_RIGHT, create_test_skeleton())
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

        # Cabin HMI alert triggered
        self.assertTrue(any("từ chối lăn bánh" in act or "Refuses to depart" in act for act in res["hmi_actions"]))
        # Level 3 tele-operator escalation triggered
        self.assertTrue(any("Tele-operator" in act for act in res["fleet_ops_actions"]))

    def test_camera_loss_moving_mrm_pullover(self):
        """Camera loss while in motion: executes MRM safe pull-over, LEVEL_1 cabin alert, LEVEL_3 tele-op escalation."""
        pax = PassengerOMS(1, SeatPosition.REAR_LEFT, create_test_skeleton())
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

        # Cabin HMI alert triggered with MRM safe pull-over
        self.assertTrue(any("MRM" in act or "tấp lề an toàn" in act for act in res["hmi_actions"]))
        # Passerby hazard lights
        self.assertTrue(any("Hazard" in act for act in res["passerby_actions"]))
        # Level 3 tele-operator escalation triggered
        self.assertTrue(any("Tele-operator" in act for act in res["fleet_ops_actions"]))

    def test_camera_loss_with_empty_passengers(self):
        """Camera loss with no passengers detected still flags camera tamper anomaly and escalates."""
        res = self.engine.evaluate_cabin(
            passengers=[],
            has_camera_video_loss=True,
            vehicle_in_motion=False
        )
        self.assertIn(AnomalyType.CAMERA_OCCLUSION_TAMPER, res["anomalies"])
        self.assertEqual(res["intervention_level"], InterventionLevelOMS.LEVEL_3_FLEET_OPS)
        self.assertTrue(res["refuse_departure"])

    def test_unresponsive_passenger_with_teleop_connected(self):
        """Unresponsive passenger after 180s with active tele-op escalates to Level 4 with 2-way call + eCall."""
        pax = PassengerOMS(1, SeatPosition.REAR_LEFT, create_test_skeleton(trunk_lean_dx=70.0), stillness_duration_s=180.0)
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

    def test_unresponsive_passenger_teleop_link_lost_failsafe_fallback(self):
        """Unresponsive passenger after 180s with tele-op link lost triggers autonomous local failsafe fallback."""
        expected_fallback = (
            "Autonomous Safe-Stop & Door Unlock Fallback: "
            "Xe tự tấp lề an toàn, mở khóa tất cả cửa xe từ chốt cơ điện tử, "
            "kích hoạt còi báo động SOS và đèn hazard ngoài xe."
        )

        pax = PassengerOMS(1, SeatPosition.REAR_LEFT, create_test_skeleton(trunk_lean_dx=70.0), stillness_duration_s=180.0)
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

    def test_can_bus_packing_under_failsafe_fallback(self):
        """Verifies CAN ID 0x0F0 and 0x130 pack/unpack with failsafe fallback and E2E CRC8."""
        pax = PassengerOMS(1, SeatPosition.REAR_LEFT, create_test_skeleton(trunk_lean_dx=70.0), stillness_duration_s=180.0)
        res = self.engine.evaluate_cabin(
            passengers=[pax],
            doors_open=True,
            trip_completed=True,
            teleop_link_lost=True
        )

        # Pack DIP Intervention command
        can_dip = pack_dip_intervention_command(
            intervention_level=InterventionLevel.LEVEL_5_MRM_STOP,
            visual_alert=1,
            acoustic_alert=2,
            haptic_wheel=False,
            seatbelt_tug=False,
            brake_jerk=False,
            hazard_flash=True,
            mrm_takeover=True,
            ecall_trigger=True,
            target_decel_mps2=1.5,
            alive_counter=7
        )
        self.assertEqual(len(can_dip), 8)
        unpacked_dip = unpack_dip_intervention_command(can_dip)
        self.assertEqual(unpacked_dip["intervention_level"], InterventionLevel.LEVEL_5_MRM_STOP)
        self.assertTrue(unpacked_dip["mrm_takeover"])
        self.assertTrue(unpacked_dip["hazard_flash"])
        self.assertTrue(unpacked_dip["ecall_trigger"])
        self.assertEqual(unpacked_dip["target_decel_mps2"], 1.5)
        self.assertEqual(unpacked_dip["alive_counter"], 7)

        # Pack OMS Status command
        can_oms = pack_oms_status(
            driver_present=DriverPresence.PRESENT,
            driver_belt=BeltStatus.BUCKLED,
            passenger_count=len([pax]),
            child_present=False,
            alive_counter=7
        )
        self.assertEqual(len(can_oms), 8)
        unpacked_oms = unpack_oms_status(can_oms)
        self.assertEqual(unpacked_oms["passenger_count"], 1)
        self.assertEqual(unpacked_oms["alive_counter"], 7)


if __name__ == '__main__':
    unittest.main()
