from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.states import (
    InterventionLevel, AttentionState, GazeZone,
    DrowsinessLevel, DriverPresence, BeltStatus
)
from src.can_e2e import E2ETracker
from src.can_frames import (
    CAN_ID_DIP_INTERVENTION, CAN_ID_DMS_STATUS, CAN_ID_OMS_STATUS,
    pack_dip_intervention_command, unpack_dip_intervention_command,
    pack_dms_status, pack_oms_status
)
from src.escalation_engine import DriverInterventionStateMachine, EscalationOutput
from src.can_bus_interface import CANBusInterface, CANMessage, get_can_physical_voltages


@dataclass
class ControllerTelemetry:
    timestamp: float
    vehicle_speed_kmh: float
    escalation: EscalationOutput
    transmitted_frame_id: int
    payload_hex: str
    alive_counter: int
    crc8: int
    is_virtual_bus: bool


class InterventionController:
    """
    Supervisory ECU coordinating DMS/OMS vision inference,
    Euro NCAP escalation state machine, and E2E-protected CAN frame transmission.
    """
    def __init__(self, can_channel: str = "vcan0"):
        self.can_bus = CANBusInterface(channel=can_channel, virtual_fallback=True)
        self.state_machine = DriverInterventionStateMachine()
        self.e2e_tracker = E2ETracker()
        self.current_time = 0.0

    def step(
        self,
        dt: float,
        vehicle_speed_kmh: float,
        gaze: GazeZone,
        attention: AttentionState,
        drowsiness: DrowsinessLevel,
        hands_on_wheel: bool,
        driver_present: DriverPresence = DriverPresence.PRESENT,
        driver_belt: BeltStatus = BeltStatus.BUCKLED
    ) -> ControllerTelemetry:
        """
        Processes one cycle:
        1. Computes escalation state.
        2. Encodes DIP_InterventionCommand CAN frame with CRC-8 and AliveCounter.
        3. Transmits over CAN_H / CAN_L physical layer.
        """
        self.current_time += dt

        # 1. Update Escalation State Machine
        esc_output = self.state_machine.update(
            dt=dt,
            vehicle_speed_kmh=vehicle_speed_kmh,
            gaze=gaze,
            attention=attention,
            drowsiness=drowsiness,
            hands_on_wheel=hands_on_wheel,
            driver_present=driver_present,
            driver_belt=driver_belt
        )

        # 2. Get next rolling alive counter
        alive_cnt = self.e2e_tracker.next_counter()

        # 3. Pack CAN Frame (ID 0x0F0)
        can_payload = pack_dip_intervention_command(
            intervention_level=esc_output.intervention_level,
            visual_alert=esc_output.visual_alert,
            acoustic_alert=esc_output.acoustic_alert,
            haptic_wheel=esc_output.haptic_wheel,
            seatbelt_tug=esc_output.seatbelt_tug,
            brake_jerk=esc_output.brake_jerk,
            hazard_flash=esc_output.hazard_flash,
            mrm_takeover=esc_output.mrm_takeover,
            ecall_trigger=esc_output.ecall_trigger,
            target_decel_mps2=esc_output.target_decel_mps2,
            alive_counter=alive_cnt
        )

        # 4. Transmit frame over CAN physical transceiver
        self.can_bus.send(CAN_ID_DIP_INTERVENTION, can_payload)

        return ControllerTelemetry(
            timestamp=self.current_time,
            vehicle_speed_kmh=vehicle_speed_kmh,
            escalation=esc_output,
            transmitted_frame_id=CAN_ID_DIP_INTERVENTION,
            payload_hex=can_payload.hex().upper(),
            alive_counter=alive_cnt,
            crc8=can_payload[7],
            is_virtual_bus=self.can_bus.is_virtual
        )

    def close(self):
        self.can_bus.close()
