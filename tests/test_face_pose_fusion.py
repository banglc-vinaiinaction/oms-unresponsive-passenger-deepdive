import unittest
import math
from src.face_pose_fusion import (
    FusedPassenger, MultimodalState, compute_hand_to_face_dist, evaluate_multimodal_passenger
)
from src.landmarks import FaceLandmarks
from src.oms_halpe import Halpe26Skeleton, Keypoint

def create_mock_landmarks(value=0.0):
    points = [(value, value) for _ in range(50)]
    return FaceLandmarks(points=points)

def create_mock_skeleton(value=0.0):
    keypoints = [Keypoint(x=value, y=value, confidence=1.0) for _ in range(26)]
    return Halpe26Skeleton(keypoints=keypoints)

class TestFacePoseFusion(unittest.TestCase):
    def test_fused_passenger_data_structure(self):
        lm = create_mock_landmarks()
        sk = create_mock_skeleton()
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk, wrist_velocity_mps=1.5)
        self.assertEqual(pax.passenger_id, 1)
        self.assertEqual(pax.wrist_velocity_mps, 1.5)
        self.assertEqual(len(pax.landmarks.points), 50)
        self.assertEqual(len(pax.skeleton.keypoints), 26)

    def test_compute_hand_to_face_dist(self):
        lm = create_mock_landmarks()
        for i in range(50):
            lm.points[i] = (10.0, 10.0)
            
        sk = create_mock_skeleton()
        sk.keypoints[9] = Keypoint(x=10.0, y=14.0, confidence=1.0)
        sk.keypoints[10] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        
        dist = compute_hand_to_face_dist(sk, lm)
        self.assertAlmostEqual(dist, 4.0)
        
    def test_zero_division_safety_guards(self):
        lm = create_mock_landmarks(0.0)
        sk = create_mock_skeleton(0.0)
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk)
        
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.NOMINAL)

    def test_multimodal_nominal(self):
        lm = create_mock_landmarks()
        for i in range(50):
            lm.points[i] = (0.0, 0.0)
        # Left eye
        lm.points[14] = (0.0, 0.0)
        lm.points[18] = (10.0, 0.0)
        lm.points[15] = (2.5, -3.0)
        lm.points[21] = (2.5, 0.0)
        lm.points[16] = (5.0, -3.0)
        lm.points[20] = (5.0, 0.0)
        lm.points[17] = (7.5, -3.0)
        lm.points[19] = (7.5, 0.0)
        # Right eye
        lm.points[22] = (20.0, 0.0)
        lm.points[26] = (30.0, 0.0)
        lm.points[23] = (22.5, -3.0)
        lm.points[29] = (22.5, 0.0)
        lm.points[24] = (25.0, -3.0)
        lm.points[28] = (25.0, 0.0)
        lm.points[25] = (27.5, -3.0)
        lm.points[27] = (27.5, 0.0)
        
        sk = create_mock_skeleton()
        sk.keypoints[18] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=15.0, y=50.0, confidence=1.0)
        
        sk.keypoints[20] = Keypoint(x=15.0, y=100.0, confidence=1.0)
        sk.keypoints[21] = Keypoint(x=15.0, y=100.0, confidence=1.0)
        sk.keypoints[24] = Keypoint(x=15.0, y=100.0, confidence=1.0)
        sk.keypoints[25] = Keypoint(x=15.0, y=100.0, confidence=1.0)
        
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.NOMINAL)

    def test_multimodal_feet_on_dashboard(self):
        lm = create_mock_landmarks()
        lm.points[14] = (0.0, 0.0); lm.points[18] = (10.0, 0.0)
        lm.points[16] = (5.0, -5.0); lm.points[20] = (5.0, 5.0)
        lm.points[22] = (20.0, 0.0); lm.points[26] = (30.0, 0.0)
        lm.points[24] = (25.0, -5.0); lm.points[28] = (25.0, 5.0)
        
        sk = create_mock_skeleton()
        sk.keypoints[18] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=15.0, y=50.0, confidence=1.0)
        sk.keypoints[20] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[21] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.FEET_ON_DASHBOARD)

    def test_multimodal_medical_emergency(self):
        lm = create_mock_landmarks()
        sk = create_mock_skeleton()
        sk.keypoints[18] = Keypoint(x=10.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=50.0, y=10.0, confidence=1.0)
        
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)
        
    def test_multimodal_motion_sickness(self):
        lm = create_mock_landmarks()
        lm.points[14] = (0.0, 0.0); lm.points[18] = (10.0, 0.0)
        lm.points[16] = (5.0, -5.0); lm.points[20] = (5.0, 5.0)
        lm.points[22] = (20.0, 0.0); lm.points[26] = (30.0, 0.0)
        lm.points[24] = (25.0, -5.0); lm.points[28] = (25.0, 5.0)
        
        lm.points[30] = (0.0, 0.0)
        lm.points[36] = (10.0, 0.0)
        lm.points[32] = (3.0, -2.5); lm.points[40] = (3.0, 2.5)
        lm.points[33] = (5.0, -2.5); lm.points[39] = (5.0, 2.5)
        lm.points[34] = (7.0, -2.5); lm.points[38] = (7.0, 2.5)
        
        for i in range(30, 42):
            lm.points[i] = (lm.points[i][0], lm.points[i][1] + 10.0)
            
        lm.points[13] = (5.0, 3.0)
        
        sk = create_mock_skeleton()
        sk.keypoints[9] = Keypoint(x=5.0, y=5.0, confidence=1.0)
        sk.keypoints[18] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=15.0, y=50.0, confidence=1.0)
        # Ensure feet are below hips (y > 50) so feet_on_dashboard is not triggered
        for f_idx in [20, 21, 22, 23, 24, 25]:
            sk.keypoints[f_idx] = Keypoint(x=15.0, y=90.0, confidence=1.0)
        
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MOTION_SICKNESS)

    def test_fallback_face_occluded_incapacitation(self):
        """When face is occluded/missing, posture trunk collapse (>40 deg) alone triggers emergency."""
        sk = create_mock_skeleton()
        sk.keypoints[18] = Keypoint(x=10.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=50.0, y=10.0, confidence=1.0) # > 40 degrees
        
        pax = FusedPassenger(passenger_id=1, landmarks=None, skeleton=sk, is_face_occluded=True)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)

    def test_fallback_face_occluded_feet_dashboard(self):
        """When face is occluded/missing, feet on dashboard is still detected via pose fallback."""
        sk = create_mock_skeleton()
        sk.keypoints[18] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=15.0, y=50.0, confidence=1.0)
        sk.keypoints[20] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[21] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        
        pax = FusedPassenger(passenger_id=1, landmarks=None, skeleton=sk, is_face_occluded=True)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.FEET_ON_DASHBOARD)

    def test_fallback_skeleton_occluded_incapacitation(self):
        """When skeleton is occluded/missing, low EAR (<0.15) + head dropped (pitch < -20) triggers emergency."""
        lm = create_mock_landmarks()
        # Closed left eye: 14 to 21
        lm.points[14] = (0.0, 0.0); lm.points[18] = (10.0, 0.0)
        lm.points[15] = (2.5, 0.0); lm.points[21] = (2.5, 0.0)
        lm.points[16] = (5.0, 0.0); lm.points[20] = (5.0, 0.0)
        lm.points[17] = (7.5, 0.0); lm.points[19] = (7.5, 0.0)

        # Closed right eye: 22 to 29
        lm.points[22] = (20.0, 0.0); lm.points[26] = (30.0, 0.0)
        lm.points[23] = (22.5, 0.0); lm.points[29] = (22.5, 0.0)
        lm.points[24] = (25.0, 0.0); lm.points[28] = (25.0, 0.0)
        lm.points[25] = (27.5, 0.0); lm.points[27] = (27.5, 0.0)

        for i in range(30, 50):
            lm.points[i] = (15.0, 100.0)
        lm.points[13] = (15.0, 90.0)
        
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=None, is_skeleton_occluded=True)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)

    def test_fallback_both_sensors_occluded(self):
        """When both cameras/modalities are occluded, returns SENSOR_OCCLUDED_FALLBACK."""
        pax = FusedPassenger(passenger_id=1, landmarks=None, skeleton=None)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.SENSOR_OCCLUDED_FALLBACK)

    def test_fallback_both_sensors_occluded_flags(self):
        """When both sensors are present as objects but flagged occluded, returns SENSOR_OCCLUDED_FALLBACK."""
        lm = create_mock_landmarks()
        sk = create_mock_skeleton()
        pax = FusedPassenger(
            passenger_id=1,
            landmarks=lm,
            skeleton=sk,
            is_face_occluded=True,
            is_skeleton_occluded=True
        )
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.SENSOR_OCCLUDED_FALLBACK)

    def test_fallback_face_occluded_violence(self):
        """When face is occluded, violence between passengers is detected via skeleton fallback."""
        sk1 = create_mock_skeleton()
        sk1.keypoints[10] = Keypoint(x=100.0, y=100.0, confidence=1.0)
        pax1 = FusedPassenger(passenger_id=1, landmarks=None, skeleton=sk1, wrist_velocity_mps=3.0, is_face_occluded=True)

        sk2 = create_mock_skeleton()
        sk2.keypoints[9] = Keypoint(x=110.0, y=110.0, confidence=1.0)
        pax2 = FusedPassenger(passenger_id=2, landmarks=None, skeleton=sk2, wrist_velocity_mps=3.0, is_face_occluded=True)

        state = evaluate_multimodal_passenger(pax1, all_passengers=[pax1, pax2])
        self.assertEqual(state, MultimodalState.HARASSMENT_VIOLENCE)

    def test_fallback_face_occluded_nominal(self):
        """When face is occluded, normal upright posture returns NOMINAL."""
        sk = create_mock_skeleton()
        sk.keypoints[18] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=15.0, y=50.0, confidence=1.0)
        sk.keypoints[20] = Keypoint(x=15.0, y=80.0, confidence=1.0)
        sk.keypoints[21] = Keypoint(x=15.0, y=80.0, confidence=1.0)
        sk.keypoints[24] = Keypoint(x=15.0, y=85.0, confidence=1.0)
        sk.keypoints[25] = Keypoint(x=15.0, y=85.0, confidence=1.0)

        pax = FusedPassenger(passenger_id=1, landmarks=None, skeleton=sk, is_face_occluded=True)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.NOMINAL)

    def test_fallback_skeleton_occluded_nominal(self):
        """When skeleton is occluded, normal alert face (eyes open, head upright) returns NOMINAL."""
        lm = create_mock_landmarks()
        # Open left eye: EAR ~ 0.33
        lm.points[14] = (0.0, 0.0); lm.points[18] = (10.0, 0.0)
        lm.points[16] = (5.0, -3.0); lm.points[20] = (5.0, 3.0)
        # Open right eye
        lm.points[22] = (20.0, 0.0); lm.points[26] = (30.0, 0.0)
        lm.points[24] = (25.0, -3.0); lm.points[28] = (25.0, 3.0)
        # Upright head pose
        lm.points[13] = (15.0, 30.0)
        for i in range(30, 50):
            lm.points[i] = (15.0, 50.0)

        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=None, is_skeleton_occluded=True)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.NOMINAL)

    def test_fallback_skeleton_occluded_pitch_drop(self):
        """When skeleton is occluded, drooping eyes (EAR < 0.2) + head dropped (pitch < -15) triggers emergency."""
        lm = create_mock_landmarks()
        # Drooping left eye (EAR ~ 0.16)
        lm.points[14] = (0.0, 0.0); lm.points[18] = (10.0, 0.0)
        lm.points[16] = (5.0, -1.6); lm.points[20] = (5.0, 1.6)
        # Drooping right eye (EAR ~ 0.16)
        lm.points[22] = (20.0, 0.0); lm.points[26] = (30.0, 0.0)
        lm.points[24] = (25.0, -1.6); lm.points[28] = (25.0, 1.6)

        # Mouth at y=100, nose at y=90 -> relative=0.9 -> pitch = -32.5 deg (< -15)
        for i in range(30, 50):
            lm.points[i] = (15.0, 100.0)
        lm.points[13] = (15.0, 90.0)

        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=None, is_skeleton_occluded=True)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MEDICAL_INCAPACITATION)

    def test_full_fusion_violence(self):
        """Full multimodal fusion detects inter-passenger violence."""
        lm1 = create_mock_landmarks()
        sk1 = create_mock_skeleton()
        sk1.keypoints[10] = Keypoint(x=100.0, y=100.0, confidence=1.0)
        pax1 = FusedPassenger(passenger_id=1, landmarks=lm1, skeleton=sk1, wrist_velocity_mps=3.2)

        lm2 = create_mock_landmarks()
        sk2 = create_mock_skeleton()
        sk2.keypoints[9] = Keypoint(x=105.0, y=105.0, confidence=1.0)
        pax2 = FusedPassenger(passenger_id=2, landmarks=lm2, skeleton=sk2, wrist_velocity_mps=2.8)

        state = evaluate_multimodal_passenger(pax1, all_passengers=[pax1, pax2])
        self.assertEqual(state, MultimodalState.HARASSMENT_VIOLENCE)

    def test_full_fusion_motion_sickness_priority_over_feet_dashboard(self):
        """In full fusion, motion sickness condition takes precedence over feet on dashboard."""
        lm = create_mock_landmarks()
        # Eyes open
        lm.points[14] = (0.0, 0.0); lm.points[18] = (10.0, 0.0)
        lm.points[16] = (5.0, -5.0); lm.points[20] = (5.0, 5.0)
        lm.points[22] = (20.0, 0.0); lm.points[26] = (30.0, 0.0)
        lm.points[24] = (25.0, -5.0); lm.points[28] = (25.0, 5.0)

        # High MAR (mouth open / grimace)
        lm.points[30] = (0.0, 0.0); lm.points[36] = (10.0, 0.0)
        lm.points[32] = (3.0, -2.5); lm.points[40] = (3.0, 2.5)
        lm.points[33] = (5.0, -2.5); lm.points[39] = (5.0, 2.5)
        lm.points[34] = (7.0, -2.5); lm.points[38] = (7.0, 2.5)
        for i in range(30, 42):
            lm.points[i] = (lm.points[i][0], lm.points[i][1] + 10.0)

        # Head tilted
        lm.points[13] = (5.0, 3.0)

        # Skeleton with hand to mouth AND feet elevated
        sk = create_mock_skeleton()
        sk.keypoints[9] = Keypoint(x=5.0, y=5.0, confidence=1.0)
        sk.keypoints[18] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[19] = Keypoint(x=15.0, y=50.0, confidence=1.0)
        sk.keypoints[20] = Keypoint(x=15.0, y=10.0, confidence=1.0)
        sk.keypoints[21] = Keypoint(x=15.0, y=10.0, confidence=1.0)

        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MOTION_SICKNESS)

if __name__ == '__main__':
    unittest.main()

