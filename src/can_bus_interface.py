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


class CANBusInterface:
    """
    Conceptual SocketCAN interface wrapper with automated loopback fallback.
    Simulates CAN frame transmission and reception across vehicle domains.
    """
    def __init__(self, channel: str = "vcan0", virtual_fallback: bool = True):
        self.channel = channel
        self.virtual_fallback = virtual_fallback
        self.sock: Optional[socket.socket] = None
        self.is_virtual = False
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
        self.sock = None

    def send(self, arbitration_id: int, payload: bytes) -> bool:
        if len(payload) > 8:
            raise ValueError(f"CAN frame payload cannot exceed 8 bytes (got {len(payload)})")
        
        # Pad payload to 8 bytes if needed
        padded_payload = payload.ljust(8, b'\x00')
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
