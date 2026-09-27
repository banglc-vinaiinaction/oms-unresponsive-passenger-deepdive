import socket
import struct
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

# Linux struct can_frame layout:
# uint32_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
# uint8_t  can_dlc; /* frame payload length in byte (0 .. 8) */
# uint8_t  __pad;   /* padding */
# uint8_t  __res0;  /* reserved / padding */
# uint8_t  __res1;  /* reserved / padding */
# uint8_t  data[8] __attribute__((aligned(8)));
CAN_FRAME_FORMAT = "=IB3x8s"


@dataclass
class CANMessage:
    arbitration_id: int
    data: bytes
    timestamp: float = 0.0
    dlc: int = 8


@dataclass
class PhysicalSignalLevel:
    bit_value: int         # 0 (Dominant) or 1 (Recessive)
    v_can_h: float         # Volts
    v_can_l: float         # Volts
    v_diff: float          # V_can_h - V_can_l (Volts)
    bus_state: str         # "DOMINANT" or "RECESSIVE"


def get_can_physical_voltages(byte_seq: bytes) -> List[PhysicalSignalLevel]:
    """
    Computes ISO 11898-2 physical line voltages across CAN_H and CAN_L twisted pair.
    - Recessive ('1'): CAN_H = 2.5V, CAN_L = 2.5V -> V_diff = 0.0V (Passive termination)
    - Dominant ('0'):  CAN_H = 3.5V, CAN_L = 1.5V -> V_diff = 2.0V (Active transceiver drive)
    """
    levels = []
    for byte in byte_seq:
        for bit_idx in range(7, -1, -1):
            bit = (byte >> bit_idx) & 1
            if bit == 0:  # Dominant
                levels.append(PhysicalSignalLevel(
                    bit_value=0,
                    v_can_h=3.5,
                    v_can_l=1.5,
                    v_diff=2.0,
                    bus_state="DOMINANT"
                ))
            else:  # Recessive
                levels.append(PhysicalSignalLevel(
                    bit_value=1,
                    v_can_h=2.5,
                    v_can_l=2.5,
                    v_diff=0.0,
                    bus_state="RECESSIVE"
                ))
    return levels


from enum import Enum, auto


class CANBusHealthState(Enum):
    HEALTHY = "HEALTHY"                     # Error-active: nominal transmission & CRC-8 verification
    PARTIAL_SHUTDOWN = "PARTIAL_SHUTDOWN"   # Error-passive/degraded: packet loss, bit-flips, or CRC corruption detected
    COMPLETE_SHUTDOWN = "COMPLETE_SHUTDOWN" # Bus-off/disconnected: total silence or physical transceiver bus-off


class CANBusInterface:
    """
    Conceptual SocketCAN interface wrapper with automated loopback fallback.
    Simulates CAN frame transmission and reception across vehicle domains with
    fault-injection capabilities (bit-flips, frame drops, and bus-off).
    """
    def __init__(self, channel: str = "vcan0", virtual_fallback: bool = True):
        self.channel = channel
        self.virtual_fallback = virtual_fallback
        self.sock: Optional[socket.socket] = None
        self.is_virtual = False
        self.is_bus_off = False
        self.bit_flip_active = False
        self.drop_frame_active = False
        self.in_memory_queue: List[CANMessage] = []
        self._init_socket()

    def _init_socket(self):
        try:
            if hasattr(socket, "AF_CAN") and hasattr(socket, "CAN_RAW"):
                self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                self.sock.bind((self.channel,))
                self.sock.setblocking(False)
                self.is_virtual = False
            else:
                self._enable_virtual_bus("AF_CAN not supported in this OS build")
        except (OSError, PermissionError) as exc:
            if self.virtual_fallback:
                self._enable_virtual_bus(f"SocketCAN interface '{self.channel}' not bound ({exc})")
            else:
                raise

    def _enable_virtual_bus(self, reason: str):
        self.is_virtual = True
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None

    def trigger_bus_off(self):
        """Simulates complete CAN controller Bus-Off state (TEC > 255)."""
        self.is_bus_off = True

    def recover_bus_off(self):
        """Recovers from Bus-Off state (simulates 128 occurrences of 11 consecutive recessive bits)."""
        self.is_bus_off = False

    def send(self, arbitration_id: int, payload: bytes) -> bool:
        if self.is_bus_off:
            # Bus-Off: Node is disconnected from physical transmission lines
            return False

        if self.drop_frame_active:
            # Simulated partial shutdown: packet drop
            return True

        if len(payload) > 8:
            raise ValueError(f"CAN frame payload cannot exceed 8 bytes (got {len(payload)})")
        
        # Pad payload to 8 bytes if needed
        padded_payload = payload.ljust(8, b'\x00')
        
        # Simulated bit flip fault injection (corrupts data byte)
        if self.bit_flip_active:
            raw_list = bytearray(padded_payload)
            raw_list[0] ^= 0xFF
            padded_payload = bytes(raw_list)

        now = time.time()
        
        if not self.is_virtual and self.sock:
            can_pkt = struct.pack(CAN_FRAME_FORMAT, arbitration_id, len(payload), padded_payload)
            try:
                self.sock.send(can_pkt)
                return True
            except OSError:
                return False
        else:
            # Virtual loopback bus
            self.in_memory_queue.append(CANMessage(
                arbitration_id=arbitration_id,
                data=padded_payload,
                timestamp=now,
                dlc=len(payload)
            ))
            return True

    def recv(self) -> Optional[CANMessage]:
        if self.is_bus_off:
            return None

        if not self.is_virtual and self.sock:
            try:
                raw_frame = self.sock.recv(16)
                if len(raw_frame) == 16:
                    can_id, can_dlc, data = struct.unpack(CAN_FRAME_FORMAT, raw_frame)
                    can_id &= 0x1FFFFFFF  # Strip EFF/RTR/ERR flags
                    return CANMessage(
                        arbitration_id=can_id,
                        data=data[:can_dlc],
                        timestamp=time.time(),
                        dlc=can_dlc
                    )
            except (BlockingIOError, socket.error):
                return None
        else:
            if self.in_memory_queue:
                return self.in_memory_queue.pop(0)
            return None

    def close(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.is_bus_off = True


class CANBusSupervisor:
    """
    Automotive Safety CAN Bus Supervisor.
    Monitors bus health across 3 defined operational integrity states:
      1. HEALTHY: Error active, sequential rolling counter, valid SAE J1850 CRC-8.
      2. PARTIAL_SHUTDOWN: Error passive / degraded, dropped frames or CRC checksum mismatch.
      3. COMPLETE_SHUTDOWN: Bus-Off state, transceiver failure, or total message silence (>250ms).
    Enforces ISO 26262 ASIL-D safe-state default actuation upon complete shutdown.
    """
    def __init__(self, timeout_threshold_s: float = 0.25):
        self.timeout_threshold_s = timeout_threshold_s
        self.total_frames_sent = 0
        self.total_frames_received = 0
        self.valid_frames = 0
        self.crc_errors = 0
        self.dropped_frames = 0
        self.last_valid_timestamp: Optional[float] = None
        self.is_bus_off_latched = False
        self.consecutive_error_counter = 0

    def record_tx(self):
        self.total_frames_sent += 1

    def record_rx_success(self, timestamp: float):
        self.total_frames_received += 1
        self.valid_frames += 1
        self.last_valid_timestamp = timestamp
        self.consecutive_error_counter = max(0, self.consecutive_error_counter - 1)

    def record_crc_error(self):
        self.total_frames_received += 1
        self.crc_errors += 1
        self.consecutive_error_counter += 8  # Simulates standard CAN REC jump (+8 per error)

    def record_frame_drop(self, count: int = 1):
        self.dropped_frames += count
        self.consecutive_error_counter += (count * 4)

    def evaluate_health(self, current_time: Optional[float] = None, bus_interface: Optional[CANBusInterface] = None) -> CANBusHealthState:
        now = current_time if current_time is not None else time.time()

        # 1. Check for hardware/simulated Bus-Off or Socket closure
        if bus_interface is not None and bus_interface.is_bus_off:
            self.is_bus_off_latched = True
            return CANBusHealthState.COMPLETE_SHUTDOWN

        # 2. Check for total silence timeout (> timeout_threshold_s)
        if self.last_valid_timestamp is not None:
            silence_duration = now - self.last_valid_timestamp
            if silence_duration >= self.timeout_threshold_s:
                self.is_bus_off_latched = True
                return CANBusHealthState.COMPLETE_SHUTDOWN

        # 3. Check for severe cumulative errors (Bus-Off threshold TEC/REC >= 256)
        if self.consecutive_error_counter >= 256 or self.is_bus_off_latched:
            self.is_bus_off_latched = True
            return CANBusHealthState.COMPLETE_SHUTDOWN

        # 4. Check for degraded / partial shutdown (CRC errors or frame drops present)
        if self.crc_errors > 0 or self.dropped_frames > 0 or self.consecutive_error_counter >= 128:
            return CANBusHealthState.PARTIAL_SHUTDOWN

        return CANBusHealthState.HEALTHY

    def get_failsafe_actuation(self) -> Dict[str, Any]:
        """
        Returns ISO 26262 ASIL-D autonomous safe-state actuator defaults:
        When the CAN bus suffers complete shutdown, downstream smart actuators
        (or hardware fail-safe supervisor relays) transition to this safe state.
        """
        return {
            "bus_health_state": CANBusHealthState.COMPLETE_SHUTDOWN.value,
            "failsafe_latched": True,
            "mrm_takeover": True,                 # Controlled deceleration to safe standstill
            "target_decel_mps2": 1.5,             # 1.5 m/s^2 gentle stop
            "hazard_flash": True,                 # Exterior emergency hazard flashers ON
            "emergency_door_unlock": True,        # Electronic door lock solenoid released
            "interior_dome_light_pct": 100,       # Cabin lights max
            "exterior_sos_beacon": True,          # Exterior acoustic speaker beacon active
            "hmi_message": "CẢNH BÁO CAN BUS-OFF: Hệ thống mất kết nối hoàn toàn. Xe tự động tấp lề & mở khóa cửa."
        }
