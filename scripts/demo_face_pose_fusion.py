import sys
import os

# Add the project root to the PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.landmarks import FaceLandmarks
from src.oms_halpe import Halpe26Skeleton, Keypoint
from src.face_pose_fusion import FusedPassenger, evaluate_multimodal_passenger, MultimodalState, compute_hand_to_face_dist
from src.dms_features import compute_ear, compute_mar, estimate_head_pose
from src.oms_halpe import compute_trunk_angle_deg

def create_base_landmarks():
    points = [(100.0, 100.0)] * 50
    # Left eye open (EAR ~ 0.33)
    points[14] = (90.0, 100.0)
    points[18] = (110.0, 100.0)
    points[16] = (100.0, 95.0)
    points[20] = (100.0, 105.0)
    points[15] = (95.0, 95.0)
    points[17] = (105.0, 95.0)
    points[19] = (105.0, 105.0)
    points[21] = (95.0, 105.0)
    
    # Right eye open
    points[22] = (130.0, 100.0)
    points[26] = (150.0, 100.0)
    points[24] = (140.0, 95.0)
    points[28] = (140.0, 105.0)
    points[23] = (135.0, 95.0)
    points[25] = (145.0, 95.0)
    points[27] = (145.0, 105.0)
    points[29] = (135.0, 105.0)
    
    # Outer lips closed (MAR ~ 0)
    points[30] = (100.0, 130.0) # left
    points[36] = (140.0, 130.0) # right
    points[33] = (120.0, 130.0) # top
    points[39] = (120.0, 130.0) # bottom
    points[32] = (110.0, 130.0)
    points[34] = (130.0, 130.0)
    points[40] = (110.0, 130.0)
    points[38] = (130.0, 130.0)
    
    # Nose (for pitch and hand-to-face)
    points[13] = (120.0, 115.0) # nose tip
    
    return points

def create_base_skeleton():
    kp = [Keypoint(200.0, 200.0, 0.9)] * 26
    kp[18] = Keypoint(120.0, 200.0, 0.9) # neck
    kp[19] = Keypoint(120.0, 400.0, 0.9) # hip_center -> trunk angle 0
    kp[9] = Keypoint(50.0, 300.0, 0.9) # left wrist
    kp[10] = Keypoint(190.0, 300.0, 0.9) # right wrist
    # Feet
    kp[20] = Keypoint(100.0, 600.0, 0.9) # left big toe
    kp[21] = Keypoint(140.0, 600.0, 0.9) # right big toe
    kp[24] = Keypoint(100.0, 620.0, 0.9) # left heel
    kp[25] = Keypoint(140.0, 620.0, 0.9) # right heel
    return kp

def print_status(pax, state, scene_name):
    ear = (compute_ear(pax.landmarks.left_eye()) + compute_ear(pax.landmarks.right_eye())) / 2
    mar = compute_mar(pax.landmarks.outer_lips(), pax.landmarks.inner_lips())
    yaw, pitch, roll = estimate_head_pose(pax.landmarks)
    trunk = compute_trunk_angle_deg(pax.skeleton)
    dist = compute_hand_to_face_dist(pax.skeleton, pax.landmarks)
    
    print(f"--- Scenario: {scene_name} ---")
    print(f"EAR: {ear:.2f} | MAR: {mar:.2f} | Head Pitch: {pitch:.1f}deg")
    print(f"Trunk: {trunk:.1f}deg | Hand-Face Dist: {dist:.1f}px")
    print(f"Pose 26-pt: Active | Face 50-pt: Active")
    print(f"CAN 0x0F0 / 0x130 State: {state.value}\n")

def test_nominal():
    pts = create_base_landmarks()
    kp = create_base_skeleton()
    pax = FusedPassenger(1, FaceLandmarks(pts), Halpe26Skeleton(kp))
    state = evaluate_multimodal_passenger(pax)
    print_status(pax, state, "NOMINAL")

def test_feet_on_dashboard():
    pts = create_base_landmarks()
    kp = create_base_skeleton()
    # Elevate feet above hip (y < hip.y - 30)
    hip_y = kp[19].y
    kp[20] = Keypoint(100.0, hip_y - 50, 0.9)
    kp[21] = Keypoint(140.0, hip_y - 50, 0.9)
    pax = FusedPassenger(1, FaceLandmarks(pts), Halpe26Skeleton(kp))
    state = evaluate_multimodal_passenger(pax)
    print_status(pax, state, "FEET_ON_DASHBOARD")

def test_motion_sickness():
    pts = create_base_landmarks()
    # High MAR (mouth open)
    pts[32] = (110.0, 110.0)
    pts[33] = (120.0, 110.0) # top lip
    pts[34] = (130.0, 110.0)
    pts[38] = (130.0, 150.0)
    pts[39] = (120.0, 150.0) # bottom lip
    pts[40] = (110.0, 150.0)
    # Head bobbing / pitch > 15
    pts[13] = (120.0, 140.0) # Move nose down to fake pitch
    
    kp = create_base_skeleton()
    # Hand to mouth
    kp[9] = Keypoint(120.0, 130.0, 0.9)
    
    pax = FusedPassenger(1, FaceLandmarks(pts), Halpe26Skeleton(kp))
    state = evaluate_multimodal_passenger(pax)
    print_status(pax, state, "MOTION_SICKNESS")

def test_medical_incapacitation():
    pts = create_base_landmarks()
    # Zero EAR
    pts[16] = (100.0, 100.0)
    pts[20] = (100.0, 100.0)
    pts[15] = (95.0, 100.0)
    pts[17] = (105.0, 100.0)
    pts[19] = (105.0, 100.0)
    pts[21] = (95.0, 100.0)
    
    pts[24] = (140.0, 100.0)
    pts[28] = (140.0, 100.0)
    pts[23] = (135.0, 100.0)
    pts[25] = (145.0, 100.0)
    pts[27] = (145.0, 100.0)
    pts[29] = (135.0, 100.0)
    
    kp = create_base_skeleton()
    # Torso collapsed
    kp[18] = Keypoint(200.0, 400.0, 0.9) # neck at same y as hip
    kp[19] = Keypoint(100.0, 400.0, 0.9)
    
    pax = FusedPassenger(1, FaceLandmarks(pts), Halpe26Skeleton(kp))
    state = evaluate_multimodal_passenger(pax)
    print_status(pax, state, "MEDICAL_INCAPACITATION")

def test_violence():
    # Pax 1
    pts1 = create_base_landmarks()
    kp1 = create_base_skeleton()
    kp1[10] = Keypoint(300.0, 300.0, 0.9) # right wrist
    pax1 = FusedPassenger(1, FaceLandmarks(pts1), Halpe26Skeleton(kp1), wrist_velocity_mps=3.0)
    
    # Pax 2
    pts2 = create_base_landmarks()
    kp2 = create_base_skeleton()
    kp2[9] = Keypoint(310.0, 310.0, 0.9) # left wrist very close to pax1 right wrist
    pax2 = FusedPassenger(2, FaceLandmarks(pts2), Halpe26Skeleton(kp2), wrist_velocity_mps=3.0)
    
    state = evaluate_multimodal_passenger(pax1, all_passengers=[pax1, pax2])
    print_status(pax1, state, "HARASSMENT_VIOLENCE")

if __name__ == "__main__":
    print("=== Robotaxi OMS Demo: Multimodal Face & Pose Fusion ===")
    test_nominal()
    test_feet_on_dashboard()
    test_motion_sickness()
    test_medical_incapacitation()
    test_violence()
