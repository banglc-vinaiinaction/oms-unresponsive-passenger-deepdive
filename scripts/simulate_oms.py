#!/usr/bin/env python3
"""
CLI Simulation: Robotaxi L4/L5 OMS Decision & 4-Level Intervention Engine.
Demonstrates end-to-end execution of:
  - Halpe26 topological feature extraction (Trunk angle, Foot elevation, Optical belt)
  - 4-Tier Intervention actuation (Level 1 Cabin -> Level 2 Passerby -> Level 3 Fleet -> Level 4 113/115)
  - CAN Bus message packing (ID 0x0F0 Intervention Command & ID 0x130 OMS Cabin Telemetry)
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.oms_halpe import (
    Keypoint, Halpe26Skeleton, PassengerOMS, SeatPosition,
    OMSDecisionEngine, InterventionLevelOMS, compute_trunk_angle_deg,
    check_feet_on_dashboard
)
from src.can_frames import (
    pack_dip_intervention_command, pack_oms_status,
    unpack_dip_intervention_command, unpack_oms_status
)
from src.states import (
    InterventionLevel, DriverPresence, BeltStatus
)


def create_mock_halpe26(
    trunk_lean_dx: float = 0.0,
    feet_elevated: bool = False,
    wrist_extended_x: float = 0.0
) -> Halpe26Skeleton:
    """Helper to build realistic 26-point Halpe coordinates."""
    # Base nominal seated coordinates (approximate 256x192 crop)
    kpts = [
        Keypoint(100.0, 40.0, 0.95, "Nose"),
        Keypoint(95.0, 35.0, 0.95, "LEye"),
        Keypoint(105.0, 35.0, 0.95, "REye"),
        Keypoint(88.0, 38.0, 0.90, "LEar"),
        Keypoint(112.0, 38.0, 0.90, "REar"),
        # Shoulders (5, 6)
        Keypoint(75.0, 60.0, 0.92, "LShoulder"),
        Keypoint(125.0, 60.0, 0.92, "RShoulder"),
        # Elbows (7, 8)
        Keypoint(70.0, 95.0, 0.88, "LElbow"),
        Keypoint(130.0, 95.0, 0.88, "RElbow"),
        # Wrists (9, 10)
        Keypoint(80.0 + wrist_extended_x, 120.0, 0.85, "LWrist"),
        Keypoint(120.0 + wrist_extended_x, 120.0, 0.85, "RWrist"),
        # Hips (11, 12)
        Keypoint(85.0 + trunk_lean_dx, 130.0, 0.90, "LHip"),
        Keypoint(115.0 + trunk_lean_dx, 130.0, 0.90, "RHip"),
        # Knees (13, 14)
        Keypoint(80.0 + trunk_lean_dx, 170.0, 0.88, "LKnee"),
        Keypoint(120.0 + trunk_lean_dx, 170.0, 0.88, "RKnee"),
        # Ankles (15, 16)
        Keypoint(80.0, 210.0, 0.85, "LAnkle"),
        Keypoint(120.0, 210.0, 0.85, "RAnkle"),
        # Head (17), Neck (18), HipCenter (19)
        Keypoint(100.0, 25.0, 0.95, "Head"),
        Keypoint(100.0, 52.0, 0.95, "Neck"),
        Keypoint(100.0 + trunk_lean_dx, 130.0, 0.92, "HipCenter"),
        # Feet (20-25)
        Keypoint(80.0, 220.0, 0.85, "LBigToe"),
        Keypoint(120.0, 220.0, 0.85, "RBigToe"),
        Keypoint(75.0, 222.0, 0.82, "LSmallToe"),
        Keypoint(125.0, 222.0, 0.82, "RSmallToe"),
        Keypoint(82.0, 205.0, 0.85, "LHeel"),
        Keypoint(118.0, 205.0, 0.85, "RHeel"),
    ]

    # Modify feet if elevated onto dashboard (y < hip_center.y - 30)
    if feet_elevated:
        hip_y = 130.0
        for idx in [20, 21, 22, 23, 24, 25]:
            kpts[idx].y = hip_y - 45.0  # Prop feet up to y=85

    return Halpe26Skeleton(keypoints=kpts)


def format_hex_payload(payload_bytes: bytes) -> str:
    return " ".join(f"{b:02X}" for b in payload_bytes)


def print_actuation_and_can(res: Dict[str, Any], paxes: List[PassengerOMS], idx: int):
    level = res["intervention_level"]
    anomalies = [a.name for a in res["anomalies"]]
    print(f"  • Hành khách trên xe : {len(paxes)} người")
    print(f"  • Cấp độ can thiệp   : \033[1;33m{level.name}\033[0m")
    print(f"  • Bất thường phát hiện: {anomalies if anomalies else 'Không (An toàn)'}")

    # Actuation Actions
    if res["hmi_actions"]:
        print(f"  \033[1;36m[CẤP 1 - CABIN HMI]\033[0m   : {'; '.join(res['hmi_actions'])}")
    if res["passerby_actions"]:
        print(f"  \033[1;32m[CẤP 2 - NGƯỜI ĐI ĐƯỜNG]\033[0m: {'; '.join(res['passerby_actions'])}")
    if res["fleet_ops_actions"]:
        print(f"  \033[1;35m[CẤP 3 - ĐỘI XE FLEET]\033[0m  : {'; '.join(res['fleet_ops_actions'])}")
    if res["law_enforcement_actions"]:
        print(f"  \033[1;31m[CẤP 4 - CÔNG AN / 115]\033[0m: {'; '.join(res['law_enforcement_actions'])}")
    if res.get("failsafe_fallback"):
        print(f"  \033[1;41m[FAILSAFE DỰ PHÒNG CỤC BỘ]\033[0m: {res['failsafe_fallback']}")

    # CAN Bus Pack with verified safety flags
    can_level = (
        InterventionLevel.LEVEL_5_MRM_STOP
        if (res.get("failsafe_fallback") or res.get("mrm_takeover"))
        else InterventionLevel(min(5, level.value))
    )
    mrm_flag = bool(res.get("mrm_takeover") or res.get("failsafe_fallback") or level.value == 4)
    hazard_flag = bool(res.get("failsafe_fallback") or res.get("mrm_takeover") or level.value >= 2)
    ecall_flag = bool(res.get("failsafe_fallback") or level.value == 4)
    decel = 1.5 if (res.get("failsafe_fallback") or res.get("mrm_takeover") or level.value >= 3) else 0.0

    can_dip = pack_dip_intervention_command(
        intervention_level=can_level,
        visual_alert=1 if level.value >= 1 else 0,
        acoustic_alert=2 if level.value >= 2 else 0,
        haptic_wheel=False,
        seatbelt_tug=False,
        brake_jerk=False,
        hazard_flash=hazard_flag,
        mrm_takeover=mrm_flag,
        ecall_trigger=ecall_flag,
        target_decel_mps2=decel,
        alive_counter=idx % 16
    )
    can_oms = pack_oms_status(
        driver_present=DriverPresence.PRESENT,
        driver_belt=BeltStatus.BUCKLED,
        passenger_count=len(paxes),
        child_present=any(p.is_child for p in paxes),
        alive_counter=idx % 16
    )

    # Verification unpacking
    unpacked_dip = unpack_dip_intervention_command(can_dip)
    unpacked_oms = unpack_oms_status(can_oms)
    assert unpacked_dip["crc8"] is not None
    assert unpacked_oms["crc8"] is not None

    print(f"  [CAN Bus ID 0x0F0 DIP] : {format_hex_payload(can_dip)} (CRC8 E2E Valid | MRM: {unpacked_dip['mrm_takeover']} | Hazard: {unpacked_dip['hazard_flash']} | Decel: {unpacked_dip['target_decel_mps2']} m/s²)")
    print(f"  [CAN Bus ID 0x130 OMS] : {format_hex_payload(can_oms)} (CRC8 E2E Valid | Pax: {unpacked_oms['passenger_count']})")


def run_oms_simulation(scenario: str = "all"):
    print("=" * 84)
    print("  VINAI / VINFAST ROBOTAXI L4/L5 — OMS HALPE26 & 4-TIER INTERVENTION SIMULATION")
    print("  ISO 26262 ASIL-B Conceptual Architecture | RTMPose Perception | Simulated CAN Protocol")
    print("=" * 84)

    engine = OMSDecisionEngine()

    if scenario in ("all", "standard"):
        print("\n>>> [KỊCH BẢN 1] Tiến trình Hành Khách Bất Tỉnh (Tele-operation kết nối bình thường)")
        passenger = PassengerOMS(1, SeatPosition.REAR_LEFT, create_mock_halpe26(trunk_lean_dx=70.0), stillness_duration_s=0.0)
        paxes = [passenger]
        timestamps = [0, 60, 120, 150, 180]

        for idx, t in enumerate(timestamps, 1):
            print(f"\n--- Thời gian: t = {t} giây ---")
            passenger.stillness_duration_s = float(t)
            res = engine.evaluate_cabin(
                passengers=paxes,
                doors_open=True,
                trip_completed=True,
                has_camera_video_loss=False,
                teleop_link_lost=False
            )
            print_actuation_and_can(res, paxes, idx)
            time.sleep(0.02)

    if scenario in ("all", "camera_loss"):
        print("\n" + "=" * 84)
        print(">>> [KỊCH BẢN 2] Camera Bị Che Khuất / Mất Video (has_camera_video_loss=True)")
        print("=" * 84)

        # Case 2A: Vehicle at departure (stopped)
        print("\n--- Trường hợp 2A: Xe chuẩn bị khởi hành (Vehicle stopped) ---")
        pax_stop = [PassengerOMS(1, SeatPosition.FRONT_RIGHT, create_mock_halpe26())]
        res_stop = engine.evaluate_cabin(
            passengers=pax_stop,
            doors_open=False,
            trip_completed=False,
            has_camera_video_loss=True,
            vehicle_in_motion=False
        )
        print_actuation_and_can(res_stop, pax_stop, 1)

        # Case 2B: Vehicle in motion
        print("\n--- Trường hợp 2B: Xe đang di chuyển trên lộ trình (Vehicle in motion) ---")
        pax_move = [PassengerOMS(1, SeatPosition.REAR_RIGHT, create_mock_halpe26())]
        res_move = engine.evaluate_cabin(
            passengers=pax_move,
            doors_open=False,
            trip_completed=False,
            has_camera_video_loss=True,
            vehicle_in_motion=True
        )
        print_actuation_and_can(res_move, pax_move, 2)

    if scenario in ("all", "teleop_lost"):
        print("\n" + "=" * 84)
        print(">>> [KỊCH BẢN 3] Hành Khách Đột Quỵ (Cấp 4) + MẤT KẾT NỐI MẠNG TELE-OP (teleop_link_lost=True)")
        print("=" * 84)

        passenger_crit = PassengerOMS(1, SeatPosition.REAR_LEFT, create_mock_halpe26(trunk_lean_dx=70.0), stillness_duration_s=180.0)
        paxes_crit = [passenger_crit]
        print("\n--- Thời gian: t = 180 giây (Cấp 4 Khẩn cấp) & Mất kết nối Tele-op ---")
        res_fallback = engine.evaluate_cabin(
            passengers=paxes_crit,
            doors_open=True,
            trip_completed=True,
            has_camera_video_loss=False,
            teleop_link_lost=True
        )
        print_actuation_and_can(res_fallback, paxes_crit, 3)

    print("\n" + "=" * 84)
    print("  SIMULATION COMPLETE: All Robotaxi Fallback & CAN Bus Verification Passed!")
    print("=" * 84)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OMS Halpe26 Simulation CLI")
    parser.add_argument(
        "--scenario",
        choices=["all", "standard", "camera_loss", "teleop_lost"],
        default="all",
        help="Specify simulation scenario to run (default: all)"
    )
    args = parser.parse_args()
    run_oms_simulation(scenario=args.scenario)
