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
        
        pax = FusedPassenger(passenger_id=1, landmarks=lm, skeleton=sk)
        state = evaluate_multimodal_passenger(pax)
        self.assertEqual(state, MultimodalState.MOTION_SICKNESS)

if __name__ == '__main__':
    unittest.main()
