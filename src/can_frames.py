import struct
from typing import Dict, Any, Tuple
from src.can_e2e import calculate_crc8_sae_j1850
from src.states import (
    InterventionLevel, AttentionState, GazeZone,
    DrowsinessLevel, DriverPresence, BeltStatus
)

CAN_ID_DIP_INTERVENTION = 0x0F0  # 240 dec
CAN_ID_DMS_STATUS        = 0x120  # 288 dec
CAN_ID_OMS_STATUS        = 0x130  # 304 dec


def pack_dip_intervention_command(
    intervention_level: InterventionLevel,
    visual_alert: int,
    acoustic_alert: int,
    haptic_wheel: bool,
    seatbelt_tug: bool,
    brake_jerk: bool,
    hazard_flash: bool,
    mrm_takeover: bool,
    ecall_trigger: bool,
    target_decel_mps2: float,
    alive_counter: int
) -> bytes:
    """
    Packs DIP_InterventionCommand (ID 0x0F0) into an 8-byte CAN frame with E2E CRC8.
    """
    b0 = (int(intervention_level) & 0x07) | ((visual_alert & 0x03) << 3) | ((acoustic_alert & 0x03) << 5) | ((1 if haptic_wheel else 0) << 7)
    b1 = ((1 if seatbelt_tug else 0) & 0x01) | \
         (((1 if brake_jerk else 0) & 0x01) << 1) | \
         (((1 if hazard_flash else 0) & 0x01) << 2) | \
         (((1 if mrm_takeover else 0) & 0x01) << 3) | \
         (((1 if ecall_trigger else 0) & 0x01) << 4)
    
    decel_raw = max(0, min(255, int(target_decel_mps2 / 0.05)))
    b2 = decel_raw
    b3 = 0x00
    b4 = 0x00
    b5 = 0x00
    b6 = alive_counter & 0x0F
    
    payload_prefix = bytes([b0, b1, b2, b3, b4, b5, b6])
    crc8 = calculate_crc8_sae_j1850(payload_prefix)
    return payload_prefix + bytes([crc8])


def unpack_dip_intervention_command(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 8:
        raise ValueError("Invalid CAN frame length for DIP_InterventionCommand")
    
    expected_crc = calculate_crc8_sae_j1850(payload[:7])
    if payload[7] != expected_crc:
        raise ValueError(f"E2E CRC check failed: expected {expected_crc:#04x}, got {payload[7]:#04x}")
    
    b0 = payload[0]
    b1 = payload[1]
    b2 = payload[2]
    b6 = payload[6]
    
    return {
        "intervention_level": InterventionLevel(b0 & 0x07),
        "visual_alert": (b0 >> 3) & 0x03,
        "acoustic_alert": (b0 >> 5) & 0x03,
        "haptic_wheel": bool((b0 >> 7) & 0x01),
        "seatbelt_tug": bool(b1 & 0x01),
        "brake_jerk": bool((b1 >> 1) & 0x01),
        "hazard_flash": bool((b1 >> 2) & 0x01),
        "mrm_takeover": bool((b1 >> 3) & 0x01),
        "ecall_trigger": bool((b1 >> 4) & 0x01),
        "target_decel_mps2": round(b2 * 0.05, 2),
        "alive_counter": b6 & 0x0F,
        "crc8": payload[7]
    }


def pack_dms_status(
    gaze_zone: GazeZone,
    attention_state: AttentionState,
    drowsiness_level: DrowsinessLevel,
    hands_on_wheel: bool,
    eye_closure_ms: int,
    perclos: float,
    alive_counter: int
) -> bytes:
    """
    Packs DMS_DriverStatus (ID 0x120) into an 8-byte CAN frame with E2E CRC8.
    """
    b0 = (int(gaze_zone) & 0x0F) | ((int(attention_state) & 0x07) << 4)
    b1 = (int(drowsiness_level) & 0x07) | ((1 if hands_on_wheel else 0) << 3)
    clamped_closure = max(0, min(65535, eye_closure_ms))
    b2 = clamped_closure & 0xFF
    b3 = (clamped_closure >> 8) & 0xFF
    b4 = max(0, min(255, int(perclos * 200)))
    b5 = 0x00
    b6 = alive_counter & 0x0F
    
    payload_prefix = bytes([b0, b1, b2, b3, b4, b5, b6])
    crc8 = calculate_crc8_sae_j1850(payload_prefix)
    return payload_prefix + bytes([crc8])


def unpack_dms_status(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 8:
        raise ValueError("Invalid CAN frame length for DMS_DriverStatus")
    
    expected_crc = calculate_crc8_sae_j1850(payload[:7])
    if payload[7] != expected_crc:
        raise ValueError(f"E2E CRC check failed: expected {expected_crc:#04x}, got {payload[7]:#04x}")
    
    b0 = payload[0]
    b1 = payload[1]
    eye_closure = payload[2] | (payload[3] << 8)
    perclos = round(payload[4] / 200.0, 3)
    b6 = payload[6]
    
    return {
        "gaze_zone": GazeZone(b0 & 0x0F),
        "attention_state": AttentionState((b0 >> 4) & 0x07),
        "drowsiness_level": DrowsinessLevel(b1 & 0x07),
        "hands_on_wheel": bool((b1 >> 3) & 0x01),
        "eye_closure_ms": eye_closure,
        "perclos": perclos,
        "alive_counter": b6 & 0x0F,
        "crc8": payload[7]
    }


def pack_oms_status(
    driver_present: DriverPresence,
    driver_belt: BeltStatus,
    passenger_count: int,
    child_present: bool,
    alive_counter: int
) -> bytes:
    """
    Packs OMS_CabinStatus (ID 0x130) into an 8-byte CAN frame with E2E CRC8.
    """
    b0 = (int(driver_present) & 0x01) | \
         ((int(driver_belt) & 0x01) << 1) | \
         ((passenger_count & 0x0F) << 2) | \
         ((1 if child_present else 0) << 6)
    payload_prefix = bytes([b0, 0x00, 0x00, 0x00, 0x00, 0x00, alive_counter & 0x0F])
    crc8 = calculate_crc8_sae_j1850(payload_prefix)
    return payload_prefix + bytes([crc8])


def unpack_oms_status(payload: bytes) -> Dict[str, Any]:
    if len(payload) != 8:
        raise ValueError("Invalid CAN frame length for OMS_CabinStatus")
    
    expected_crc = calculate_crc8_sae_j1850(payload[:7])
    if payload[7] != expected_crc:
        raise ValueError(f"E2E CRC check failed: expected {expected_crc:#04x}, got {payload[7]:#04x}")
    
    b0 = payload[0]
    return {
        "driver_present": DriverPresence(b0 & 0x01),
        "driver_belt_fastened": BeltStatus((b0 >> 1) & 0x01),
        "passenger_count": (b0 >> 2) & 0x0F,
        "child_present": bool((b0 >> 6) & 0x01),
        "alive_counter": payload[6] & 0x0F,
        "crc8": payload[7]
    }
