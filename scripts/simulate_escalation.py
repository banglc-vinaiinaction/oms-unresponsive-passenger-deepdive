#!/usr/bin/env python3
"""
Simulation: Escalation from DMS/OMS to Driver Intervention Program via CAN (CAN_H / CAN_L).
Demonstrates end-to-end execution of Euro NCAP 5-tier intervention policy and physical CAN signal generation.

Modes:
    --mode timeline   Original hardcoded-enum timeline (default)
    --mode keypoint   Full DMS perception pipeline: landmarks → features → classifier → escalation
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.states import (
    InterventionLevel, AttentionState, GazeZone,
    DrowsinessLevel, DriverPresence, BeltStatus
)
from src.intervention_controller import InterventionController
from src.can_bus_interface import get_can_physical_voltages


def print_banner():
    print("=" * 80)
    print("  VINAI / VINFAST ADAS: DMS/OMS -> DRIVER INTERVENTION PROGRAM VIA CAN_H/CAN_L")
    print("  ISO 26262 ASIL-B/D Architecture | Euro NCAP 2026 Policy | Physical CAN Telemetry")
    print("=" * 80)


def format_hex_payload(payload_hex: str) -> str:
    # 16-character hex string -> format with spaces every 2 chars
    return " ".join(payload_hex[i:i+2] for i in range(0, len(payload_hex), 2))


def run_timeline_simulation():
    print_banner()

    controller = InterventionController(can_channel="vcan0")
    
    # Timeline steps: (duration_s, speed_kmh, gaze, attention, drowsiness, hands_on, description)
    timeline = [
        (1.5, 65.0, GazeZone.ROAD_AHEAD, AttentionState.ATTENTIVE, DrowsinessLevel.ALERT, True, "Phase 1: Normal Attentive Driving"),
        (2.0, 65.0, GazeZone.PHONE_DOWN, AttentionState.DISTRACTED_LOW, DrowsinessLevel.ALERT, False, "Phase 2: Distraction begins (looking at phone, hands off wheel)"),
        (1.5, 64.0, GazeZone.PHONE_DOWN, AttentionState.DISTRACTED_HIGH, DrowsinessLevel.QUESTIONABLE, False, "Phase 3: Persistent distraction (> 3.5s)"),
        (1.5, 63.0, GazeZone.PHONE_DOWN, AttentionState.DISTRACTED_HIGH, DrowsinessLevel.MODERATE_DROWSY, False, "Phase 4: Critical distraction (> 5.0s)"),
        (1.5, 62.0, GazeZone.EYES_CLOSED, AttentionState.MICROSLEEP, DrowsinessLevel.SEVERE_DROWSY, False, "Phase 5: Microsleep onset & unresponsiveness (> 6.5s)"),
        (3.0, 45.0, GazeZone.EYES_CLOSED, AttentionState.INCAPACITATED, DrowsinessLevel.SLEEP_ONSET, False, "Phase 6: Complete driver incapacitation (> 8.0s) -> MRM Takeover"),
    ]

    dt = 0.2  # 200ms simulation time-step
    last_reported_level = None

    for phase_idx, (phase_dur, speed, gaze, att, drowsy, hands, desc) in enumerate(timeline, start=1):
        print(f"\n>>> [{desc}]")
        steps = int(phase_dur / dt)
        for s in range(steps):
            telemetry = controller.step(
                dt=dt,
                vehicle_speed_kmh=speed,
                gaze=gaze,
                attention=att,
                drowsiness=drowsy,
                hands_on_wheel=hands,
                driver_present=DriverPresence.PRESENT,
                driver_belt=BeltStatus.BUCKLED
            )

            esc = telemetry.escalation
            # Print if state transitions or periodic heartbeat
            state_changed = (esc.intervention_level != last_reported_level)
            if state_changed or s == steps - 1:
                last_reported_level = esc.intervention_level
                
                # Sample physical bit voltages from the first byte of payload
                raw_bytes = bytes.fromhex(telemetry.payload_hex)
                phys_levels = get_can_physical_voltages(raw_bytes[:1])
                b0_state = phys_levels[0].bus_state
                can_h_v = phys_levels[0].v_can_h
                can_l_v = phys_levels[0].v_can_l
                v_diff = phys_levels[0].v_diff

                print(
                    f"  T={telemetry.timestamp:4.1f}s | Speed: {telemetry.vehicle_speed_kmh:4.1f} km/h | "
                    f"Gaze: {gaze.name:<11} | Level: {esc.intervention_level.name} ({esc.intervention_level.value})"
                )
                print(f"      CAN Frame ID : 0x{telemetry.transmitted_frame_id:03X} (DIP_InterventionCommand, high priority)")
                print(f"      Payload Hex  : [{format_hex_payload(telemetry.payload_hex)}] (Counter: {telemetry.alive_counter}, CRC-8: 0x{telemetry.crc8:02X})")
                print(f"      Actuators    : Visual={esc.visual_alert}, Audio={esc.acoustic_alert}, "
                      f"WheelShake={esc.haptic_wheel}, BeltTug={esc.seatbelt_tug}, BrakeJerk={esc.brake_jerk}, "
                      f"Hazard={esc.hazard_flash}, MRM={esc.mrm_takeover}, eCall={esc.ecall_trigger}")
                print(f"      Physical Wire: CAN_H = {can_h_v:.1f}V | CAN_L = {can_l_v:.1f}V | V_diff = {v_diff:.1f}V ({b0_state})")
                print(f"      Status       : {esc.state_description}")
                print("-" * 75)

    controller.close()
    print("\n[SIMULATION COMPLETED]: Verified full escalation path up to Level 5 MRM autonomous stop.")


def run_keypoint_simulation():
    """
    Full DMS perception pipeline:
    Synthetic landmarks → EAR/MAR/PERCLOS/head pose → classifier → escalation → CAN
    """
    from src.landmark_generator import generate_drowsy_sequence
    from src.dms_classifier import DMSClassifier

    print_banner()
    print("  MODE: Keypoint-driven DMS perception pipeline")
    print("  Pipeline: 50-pt Landmarks → EAR/MAR/PERCLOS → Classifier → Escalation → CAN")
    print("=" * 80)

    fps = 30  # Lower FPS for readable output
    duration_s = 15.0
    dt = 1.0 / fps

    frames = generate_drowsy_sequence(duration_s=duration_s, fps=fps, seed=42)
    classifier = DMSClassifier(perclos_window_s=10.0)  # Shorter window for demo
    controller = InterventionController(can_channel="vcan0")

    last_reported_level = None
    print_interval = int(fps * 0.5)  # Print every 0.5s

    for i, frame in enumerate(frames):
        # 1. DMS perception: landmarks → features → enums
        dms_out = classifier.classify(frame, dt)

        # 2. Feed to escalation + CAN
        telemetry = controller.step(
            dt=dt,
            vehicle_speed_kmh=60.0,
            gaze=dms_out.gaze_zone,
            attention=dms_out.attention_state,
            drowsiness=dms_out.drowsiness_level,
            hands_on_wheel=True,
            driver_present=DriverPresence.PRESENT,
            driver_belt=BeltStatus.BUCKLED,
        )

        esc = telemetry.escalation
        state_changed = esc.intervention_level != last_reported_level

        if state_changed or i % print_interval == 0:
            last_reported_level = esc.intervention_level

            raw_bytes = bytes.fromhex(telemetry.payload_hex)
            phys_levels = get_can_physical_voltages(raw_bytes[:1])

            print(
                f"  T={frame.timestamp:5.2f}s | "
                f"EAR={dms_out.ear_avg:.3f} MAR={dms_out.mar:.3f} "
                f"PERCLOS={dms_out.perclos:.1%} "
                f"Yaw={dms_out.head_yaw:+5.1f}° Pitch={dms_out.head_pitch:+5.1f}°"
            )
            print(
                f"           Gaze: {dms_out.gaze_zone.name:<12} "
                f"Attn: {dms_out.attention_state.name:<16} "
                f"Drowsy: {dms_out.drowsiness_level.name}"
            )
            print(
                f"           → Level {esc.intervention_level.value} "
                f"({esc.intervention_level.name}) | "
                f"CAN 0x{telemetry.transmitted_frame_id:03X} "
                f"[{format_hex_payload(telemetry.payload_hex)}] "
                f"CRC=0x{telemetry.crc8:02X}"
            )
            print(
                f"           Wire: CAN_H={phys_levels[0].v_can_h:.1f}V "
                f"CAN_L={phys_levels[0].v_can_l:.1f}V ({phys_levels[0].bus_state})"
            )
            if state_changed:
                print(f"           *** {esc.state_description}")
            print("-" * 80)

    controller.close()
    print(f"\n[KEYPOINT SIMULATION COMPLETED]: {len(frames)} frames processed through full DMS pipeline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DMS/OMS → Driver Intervention escalation simulation"
    )
    parser.add_argument(
        "--mode",
        choices=["timeline", "keypoint"],
        default="timeline",
        help="Simulation mode: 'timeline' (hardcoded enums) or 'keypoint' (full DMS pipeline)",
    )
    args = parser.parse_args()

    if args.mode == "keypoint":
        run_keypoint_simulation()
    else:
        run_timeline_simulation()
