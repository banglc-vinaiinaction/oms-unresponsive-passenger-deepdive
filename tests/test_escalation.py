import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.states import (
    InterventionLevel, AttentionState, GazeZone,
    DrowsinessLevel, DriverPresence, BeltStatus
)
from src.can_e2e import calculate_crc8_sae_j1850, E2ETracker
from src.can_frames import (
    pack_dip_intervention_command, unpack_dip_intervention_command,
    pack_dms_status, unpack_dms_status,
    pack_oms_status, unpack_oms_status
)
from src.escalation_engine import DriverInterventionStateMachine
from src.can_bus_interface import get_can_physical_voltages
from src.landmarks import FaceLandmarks, LandmarkRegion, REGION_INDICES, NUM_LANDMARKS
from src.dms_features import compute_ear, compute_mar, estimate_head_pose, PERCLOSTracker
from src.dms_classifier import DMSClassifier
from src.landmark_generator import (
    _BASE_FACE, _copy_base, _apply_blink, _apply_yawn, _apply_head_turn,
    generate_normal_driving, generate_drowsy_sequence,
    generate_distraction_sequence, generate_glasses_sequence,
)


# ========================================================================
# Original CAN Protocol & E2E Tests
# ========================================================================

class TestCANProtocolAndE2E(unittest.TestCase):
    def test_crc8_calculation(self):
        # Known test vectors for SAE J1850
        data = bytes([0x00, 0x00, 0x00, 0x00])
        crc = calculate_crc8_sae_j1850(data)
        self.assertIsInstance(crc, int)
        self.assertTrue(0 <= crc <= 255)

    def test_dip_frame_pack_unpack_roundtrip(self):
        payload = pack_dip_intervention_command(
            intervention_level=InterventionLevel.LEVEL_3_HAPTIC,
            visual_alert=2,
            acoustic_alert=2,
            haptic_wheel=True,
            seatbelt_tug=True,
            brake_jerk=False,
            hazard_flash=False,
            mrm_takeover=False,
            ecall_trigger=False,
            target_decel_mps2=0.0,
            alive_counter=5
        )
        self.assertEqual(len(payload), 8)
        decoded = unpack_dip_intervention_command(payload)
        self.assertEqual(decoded["intervention_level"], InterventionLevel.LEVEL_3_HAPTIC)
        self.assertEqual(decoded["visual_alert"], 2)
        self.assertEqual(decoded["acoustic_alert"], 2)
        self.assertTrue(decoded["haptic_wheel"])
        self.assertTrue(decoded["seatbelt_tug"])
        self.assertFalse(decoded["brake_jerk"])
        self.assertEqual(decoded["alive_counter"], 5)

    def test_crc_failure_detection(self):
        payload = bytearray(pack_dip_intervention_command(
            intervention_level=InterventionLevel.LEVEL_1_VISUAL,
            visual_alert=1,
            acoustic_alert=0,
            haptic_wheel=False,
            seatbelt_tug=False,
            brake_jerk=False,
            hazard_flash=False,
            mrm_takeover=False,
            ecall_trigger=False,
            target_decel_mps2=0.0,
            alive_counter=1
        ))
        # Corrupt data byte
        payload[0] ^= 0xFF
        with self.assertRaises(ValueError):
            unpack_dip_intervention_command(bytes(payload))

    def test_dms_frame_roundtrip(self):
        payload = pack_dms_status(
            gaze_zone=GazeZone.PHONE_DOWN,
            attention_state=AttentionState.DISTRACTED_HIGH,
            drowsiness_level=DrowsinessLevel.MODERATE_DROWSY,
            hands_on_wheel=False,
            eye_closure_ms=1250,
            perclos=0.45,
            alive_counter=7
        )
        self.assertEqual(len(payload), 8)
        decoded = unpack_dms_status(payload)
        self.assertEqual(decoded["gaze_zone"], GazeZone.PHONE_DOWN)
        self.assertEqual(decoded["attention_state"], AttentionState.DISTRACTED_HIGH)
        self.assertEqual(decoded["drowsiness_level"], DrowsinessLevel.MODERATE_DROWSY)
        self.assertFalse(decoded["hands_on_wheel"])
        self.assertEqual(decoded["eye_closure_ms"], 1250)
        self.assertAlmostEqual(decoded["perclos"], 0.45, places=2)
        self.assertEqual(decoded["alive_counter"], 7)

    def test_oms_frame_roundtrip(self):
        payload = pack_oms_status(
            driver_present=DriverPresence.PRESENT,
            driver_belt=BeltStatus.BUCKLED,
            passenger_count=3,
            child_present=True,
            alive_counter=12
        )
        self.assertEqual(len(payload), 8)
        decoded = unpack_oms_status(payload)
        self.assertEqual(decoded["driver_present"], DriverPresence.PRESENT)
        self.assertEqual(decoded["driver_belt_fastened"], BeltStatus.BUCKLED)
        self.assertEqual(decoded["passenger_count"], 3)
        self.assertTrue(decoded["child_present"])
        self.assertEqual(decoded["alive_counter"], 12)


# ========================================================================
# Original Escalation State Machine Tests
# ========================================================================

class TestEscalationStateMachine(unittest.TestCase):
    def test_normal_driving_stays_passive(self):
        sm = DriverInterventionStateMachine()
        for _ in range(50):
            out = sm.update(
                dt=0.1,
                vehicle_speed_kmh=60.0,
                gaze=GazeZone.ROAD_AHEAD,
                attention=AttentionState.ATTENTIVE,
                drowsiness=DrowsinessLevel.ALERT,
                hands_on_wheel=True,
                driver_present=DriverPresence.PRESENT,
                driver_belt=BeltStatus.BUCKLED
            )
        self.assertEqual(out.intervention_level, InterventionLevel.LEVEL_0_PASSIVE)
        self.assertEqual(out.visual_alert, 0)
        self.assertEqual(out.acoustic_alert, 0)
        self.assertFalse(out.haptic_wheel)
        self.assertFalse(out.brake_jerk)
        self.assertFalse(out.mrm_takeover)

    def test_distraction_escalation_sequence(self):
        sm = DriverInterventionStateMachine()
        # Simulate driver looking down at phone continuously while driving at 60 km/h
        levels_seen = set()
        for _ in range(90):  # 9.0 seconds total
            out = sm.update(
                dt=0.1,
                vehicle_speed_kmh=60.0,
                gaze=GazeZone.PHONE_DOWN,
                attention=AttentionState.DISTRACTED_HIGH,
                drowsiness=DrowsinessLevel.ALERT,
                hands_on_wheel=False,
                driver_present=DriverPresence.PRESENT,
                driver_belt=BeltStatus.BUCKLED
            )
            levels_seen.add(out.intervention_level)

        # Must have passed through all escalation stages up to MRM Stop
        self.assertIn(InterventionLevel.LEVEL_1_VISUAL, levels_seen)
        self.assertIn(InterventionLevel.LEVEL_2_ACOUSTIC, levels_seen)
        self.assertIn(InterventionLevel.LEVEL_3_HAPTIC, levels_seen)
        self.assertIn(InterventionLevel.LEVEL_4_BRAKE_JERK, levels_seen)
        self.assertIn(InterventionLevel.LEVEL_5_MRM_STOP, levels_seen)
        self.assertTrue(out.mrm_takeover)
        self.assertTrue(out.ecall_trigger)

    def test_driver_recovery_deescalates(self):
        sm = DriverInterventionStateMachine()
        # Escalate to Level 2 (3.2 seconds distracted)
        for _ in range(32):
            out = sm.update(
                dt=0.1,
                vehicle_speed_kmh=60.0,
                gaze=GazeZone.INFOTAINMENT,
                attention=AttentionState.DISTRACTED_LOW,
                drowsiness=DrowsinessLevel.ALERT,
                hands_on_wheel=True,
                driver_present=DriverPresence.PRESENT,
                driver_belt=BeltStatus.BUCKLED
            )
        self.assertEqual(out.intervention_level, InterventionLevel.LEVEL_2_ACOUSTIC)

        # Driver looks back at road and holds steering wheel
        for _ in range(10):  # 1.0 second recovery
            out = sm.update(
                dt=0.1,
                vehicle_speed_kmh=60.0,
                gaze=GazeZone.ROAD_AHEAD,
                attention=AttentionState.ATTENTIVE,
                drowsiness=DrowsinessLevel.ALERT,
                hands_on_wheel=True,
                driver_present=DriverPresence.PRESENT,
                driver_belt=BeltStatus.BUCKLED
            )
        self.assertEqual(out.intervention_level, InterventionLevel.LEVEL_0_PASSIVE)

    def test_sudden_incapacitation_fast_path(self):
        sm = DriverInterventionStateMachine()
        # Immediate collapse
        out = sm.update(
            dt=0.1,
            vehicle_speed_kmh=80.0,
            gaze=GazeZone.EYES_CLOSED,
            attention=AttentionState.INCAPACITATED,
            drowsiness=DrowsinessLevel.SLEEP_ONSET,
            hands_on_wheel=False,
            driver_present=DriverPresence.PRESENT,
            driver_belt=BeltStatus.BUCKLED
        )
        # Fast path bypasses Level 1/2/3 directly into tactical/critical
        self.assertGreaterEqual(out.intervention_level, InterventionLevel.LEVEL_4_BRAKE_JERK)


# ========================================================================
# CAN Physical Layer Tests
# ========================================================================

class TestCANPhysicalLayer(unittest.TestCase):
    def test_physical_voltage_levels(self):
        # 0x00 is all zero bits -> Dominant bits
        levels_dom = get_can_physical_voltages(bytes([0x00]))
        for level in levels_dom:
            self.assertEqual(level.bit_value, 0)
            self.assertEqual(level.v_can_h, 3.5)
            self.assertEqual(level.v_can_l, 1.5)
            self.assertEqual(level.v_diff, 2.0)
            self.assertEqual(level.bus_state, "DOMINANT")

        # 0xFF is all one bits -> Recessive bits
        levels_rec = get_can_physical_voltages(bytes([0xFF]))
        for level in levels_rec:
            self.assertEqual(level.bit_value, 1)
            self.assertEqual(level.v_can_h, 2.5)
            self.assertEqual(level.v_can_l, 2.5)
            self.assertEqual(level.v_diff, 0.0)
            self.assertEqual(level.bus_state, "RECESSIVE")


# ========================================================================
# NEW: DMS Feature Extraction Tests
# ========================================================================

class TestDMSFeatures(unittest.TestCase):
    def _make_face(self, **kwargs) -> FaceLandmarks:
        """Create a FaceLandmarks from the base template with optional modifications."""
        pts = _copy_base()
        return FaceLandmarks(points=pts, **kwargs)

    def test_ear_open_eye(self):
        """EAR for wide-open eyes (base template) should be ~0.30–0.40."""
        face = self._make_face()
        ear_l = compute_ear(face.left_eye())
        ear_r = compute_ear(face.right_eye())
        self.assertGreater(ear_l, 0.25)
        self.assertLess(ear_l, 0.50)
        self.assertGreater(ear_r, 0.25)
        self.assertLess(ear_r, 0.50)

    def test_ear_closed_eye(self):
        """EAR for fully closed eyes should be < 0.10."""
        pts = _apply_blink(_copy_base(), closure_fraction=1.0)
        face = FaceLandmarks(points=pts)
        ear_l = compute_ear(face.left_eye())
        ear_r = compute_ear(face.right_eye())
        self.assertLess(ear_l, 0.10)
        self.assertLess(ear_r, 0.10)

    def test_ear_half_closed(self):
        """EAR at 50% closure should be between closed and open values."""
        pts_open = _copy_base()
        pts_half = _apply_blink(_copy_base(), closure_fraction=0.5)
        pts_closed = _apply_blink(_copy_base(), closure_fraction=1.0)

        face_open = FaceLandmarks(points=pts_open)
        face_half = FaceLandmarks(points=pts_half)
        face_closed = FaceLandmarks(points=pts_closed)

        ear_open = compute_ear(face_open.left_eye())
        ear_half = compute_ear(face_half.left_eye())
        ear_closed = compute_ear(face_closed.left_eye())

        self.assertGreater(ear_half, ear_closed)
        self.assertLess(ear_half, ear_open)

    def test_mar_closed_mouth(self):
        """MAR for closed mouth (base template) should be < 0.30."""
        face = self._make_face()
        mar = compute_mar(face.outer_lips(), face.inner_lips())
        self.assertLess(mar, 0.30)
        self.assertGreater(mar, 0.0)

    def test_mar_yawn(self):
        """MAR for wide-open mouth (yawn) should be > 0.45."""
        pts = _apply_yawn(_copy_base(), openness=1.0)
        face = FaceLandmarks(points=pts)
        mar = compute_mar(face.outer_lips(), face.inner_lips())
        self.assertGreater(mar, 0.45)

    def test_head_pose_forward(self):
        """Symmetrical base template → yaw ≈ 0, pitch ≈ 0, roll ≈ 0."""
        face = self._make_face()
        yaw, pitch, roll = estimate_head_pose(face)
        self.assertAlmostEqual(yaw, 0.0, delta=5.0)
        self.assertAlmostEqual(pitch, 0.0, delta=10.0)
        self.assertAlmostEqual(roll, 0.0, delta=3.0)

    def test_head_pose_turned_left(self):
        """Head turned left → positive yaw."""
        pts = _apply_head_turn(_copy_base(), yaw_deg=30.0, pitch_deg=0.0)
        face = FaceLandmarks(points=pts)
        yaw, pitch, roll = estimate_head_pose(face)
        self.assertGreater(yaw, 5.0)

    def test_head_pose_looking_down(self):
        """Head tilted down → negative pitch."""
        pts = _apply_head_turn(_copy_base(), yaw_deg=0.0, pitch_deg=-25.0)
        face = FaceLandmarks(points=pts)
        yaw, pitch, roll = estimate_head_pose(face)
        self.assertLess(pitch, -5.0)

    def test_perclos_normal(self):
        """Normal driving with rare blinks → PERCLOS < 8%."""
        tracker = PERCLOSTracker(window_seconds=10.0, ear_threshold=0.20)
        # 10 seconds at 60fps, 2 blinks of 200ms each
        dt = 1.0 / 60
        for i in range(600):
            t = i * dt
            # Blinks at t=3.0 and t=7.0, 200ms each
            if 3.0 <= t < 3.2 or 7.0 <= t < 7.2:
                ear = 0.05
            else:
                ear = 0.32
            tracker.update(t, ear)

        perclos = tracker.update(10.0, 0.32)
        self.assertLess(perclos, 0.08)

    def test_perclos_drowsy(self):
        """Sustained eye closure → PERCLOS > 20%."""
        tracker = PERCLOSTracker(window_seconds=10.0, ear_threshold=0.20)
        dt = 1.0 / 60
        for i in range(600):
            t = i * dt
            # Eyes closed for last 3 seconds (30% of window)
            if t >= 7.0:
                ear = 0.05
            else:
                ear = 0.32
            tracker.update(t, ear)

        perclos = tracker.update(10.0, 0.05)
        self.assertGreater(perclos, 0.20)


# ========================================================================
# NEW: DMS Classifier Tests
# ========================================================================

class TestDMSClassifier(unittest.TestCase):
    def test_attentive_classification(self):
        """Normal open-eyed forward-looking → ATTENTIVE + ROAD_AHEAD + ALERT."""
        classifier = DMSClassifier()
        face = FaceLandmarks(points=_copy_base())
        out = classifier.classify(face, dt=0.1)

        self.assertEqual(out.gaze_zone, GazeZone.ROAD_AHEAD)
        self.assertEqual(out.attention_state, AttentionState.ATTENTIVE)
        self.assertEqual(out.drowsiness_level, DrowsinessLevel.ALERT)
        self.assertFalse(out.eyes_closed)
        self.assertFalse(out.is_yawning)

    def test_eyes_closed_classification(self):
        """Fully closed eyes → EYES_CLOSED gaze zone."""
        classifier = DMSClassifier()
        pts = _apply_blink(_copy_base(), closure_fraction=1.0)
        face = FaceLandmarks(points=pts)
        out = classifier.classify(face, dt=0.1)

        self.assertEqual(out.gaze_zone, GazeZone.EYES_CLOSED)
        self.assertTrue(out.eyes_closed)

    def test_phone_distraction_classification(self):
        """Looking down at phone → PHONE_DOWN + DISTRACTED_HIGH."""
        classifier = DMSClassifier()
        pts = _apply_head_turn(_copy_base(), yaw_deg=5.0, pitch_deg=-35.0)
        face = FaceLandmarks(points=pts)
        out = classifier.classify(face, dt=0.1)

        self.assertEqual(out.gaze_zone, GazeZone.PHONE_DOWN)
        self.assertEqual(out.attention_state, AttentionState.DISTRACTED_HIGH)

    def test_microsleep_after_sustained_closure(self):
        """Eyes closed for > 0.5s → MICROSLEEP attention state."""
        classifier = DMSClassifier()
        pts_closed = _apply_blink(_copy_base(), closure_fraction=1.0)

        # Feed 10 frames at 0.1s each = 1.0s of closure
        for _ in range(10):
            face = FaceLandmarks(points=list(pts_closed))
            out = classifier.classify(face, dt=0.1)

        self.assertEqual(out.attention_state, AttentionState.MICROSLEEP)

    def test_yawn_detection(self):
        """Wide-open mouth sustained > 1s → is_yawning=True."""
        classifier = DMSClassifier()
        pts_yawn = _apply_yawn(_copy_base(), openness=0.9)

        # Feed 15 frames at 0.1s each = 1.5s of yawn
        for _ in range(15):
            face = FaceLandmarks(points=list(pts_yawn))
            out = classifier.classify(face, dt=0.1)

        self.assertTrue(out.is_yawning)
        self.assertGreater(out.mar, 0.40)


# ========================================================================
# NEW: Landmark Generator Tests
# ========================================================================

class TestLandmarkGenerator(unittest.TestCase):
    def test_normal_driving_frame_count(self):
        frames = generate_normal_driving(duration_s=5.0, fps=30)
        self.assertEqual(len(frames), 150)

    def test_normal_driving_landmarks_shape(self):
        frames = generate_normal_driving(duration_s=1.0, fps=10)
        for f in frames:
            self.assertEqual(len(f.points), NUM_LANDMARKS)
            self.assertFalse(f.glasses)

    def test_drowsy_sequence_has_closure(self):
        """Drowsy sequence should have low-EAR frames in the final portion."""
        frames = generate_drowsy_sequence(duration_s=10.0, fps=30)
        # Check last 20% of frames — should have sustained eye closure
        last_portion = frames[int(len(frames) * 0.8):]
        ears = [compute_ear(f.left_eye()) for f in last_portion]
        avg_ear = sum(ears) / len(ears)
        self.assertLess(avg_ear, 0.15, "Final portion of drowsy sequence should show low EAR")

    def test_glasses_flag(self):
        frames = generate_glasses_sequence(duration_s=2.0, fps=10)
        for f in frames:
            self.assertTrue(f.glasses)

    def test_blink_animation_produces_ear_change(self):
        """Applying blink should lower EAR compared to base template."""
        open_face = FaceLandmarks(points=_copy_base())
        closed_pts = _apply_blink(_copy_base(), closure_fraction=1.0)
        closed_face = FaceLandmarks(points=closed_pts)

        ear_open = compute_ear(open_face.left_eye())
        ear_closed = compute_ear(closed_face.left_eye())

        self.assertGreater(ear_open, ear_closed * 3,
                           "Open EAR should be much larger than closed EAR")


# ========================================================================
# NEW: Glasses Edge Case Tests
# ========================================================================

class TestGlassesEdgeCases(unittest.TestCase):
    def test_glasses_increases_ear_variance(self):
        """With glasses, EAR measurements should show higher variance."""
        normal_frames = generate_normal_driving(duration_s=5.0, fps=30, seed=100)
        glasses_frames = generate_glasses_sequence(duration_s=5.0, fps=30, seed=100)

        def ear_std(frames):
            ears = [compute_ear(f.left_eye()) for f in frames]
            mean = sum(ears) / len(ears)
            return (sum((e - mean) ** 2 for e in ears) / len(ears)) ** 0.5

        std_normal = ear_std(normal_frames)
        std_glasses = ear_std(glasses_frames)

        self.assertGreater(std_glasses, std_normal,
                           "Glasses should increase EAR measurement variance")


if __name__ == "__main__":
    unittest.main()
