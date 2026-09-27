from dataclasses import dataclass
from src.states import (
    InterventionLevel, AttentionState, GazeZone,
    DrowsinessLevel, DriverPresence, BeltStatus
)

@dataclass
class EscalationOutput:
    intervention_level: InterventionLevel
    visual_alert: int          # 0=None, 1=Yellow, 2=Red Flashing
    acoustic_alert: int        # 0=None, 1=Soft chime, 2=Urgent alarm (80dB)
    haptic_wheel: bool         # Steering vibration motor
    seatbelt_tug: bool         # Reversible motorized pretensioner jerk
    brake_jerk: bool           # 300ms deceleration pulse (-2.5 m/s^2)
    hazard_flash: bool         # Hazard lights request
    mrm_takeover: bool         # Autonomous Minimum Risk Maneuver takeover
    ecall_trigger: bool        # Automated emergency call trigger
    target_decel_mps2: float   # Commanded deceleration
    state_description: str     # Operational status description


class DriverInterventionStateMachine:
    """
    Deterministic safety state machine implementing Euro NCAP 2026 / ISO 26262 
    driver intervention escalation via CAN bus.
    """
    def __init__(self):
        self.current_level = InterventionLevel.LEVEL_0_PASSIVE
        self.distraction_timer = 0.0
        self.eye_closure_timer = 0.0
        self.hands_off_timer = 0.0
        self.brake_jerk_remaining = 0.0
        self.recovery_timer = 0.0
        self.mrm_latched = False

    def is_gaze_attentive(self, gaze: GazeZone) -> bool:
        return gaze in (GazeZone.ROAD_AHEAD, GazeZone.LEFT_MIRROR, GazeZone.RIGHT_MIRROR, GazeZone.CLUSTER)

    def update(
        self,
        dt: float,
        vehicle_speed_kmh: float,
        gaze: GazeZone,
        attention: AttentionState,
        drowsiness: DrowsinessLevel,
        hands_on_wheel: bool,
        driver_present: DriverPresence,
        driver_belt: BeltStatus
    ) -> EscalationOutput:
        """
        Executes one discrete state evaluation tick (typically 10ms - 20ms).
        """
        # If vehicle is stationary and not latched in MRM, suppress critical dynamic intervention
        is_moving = vehicle_speed_kmh > 1.5

        # 1. Update timers
        if not self.is_gaze_attentive(gaze) or attention in (AttentionState.DISTRACTED_LOW, AttentionState.DISTRACTED_HIGH):
            self.distraction_timer += dt
        else:
            self.distraction_timer = max(0.0, self.distraction_timer - (dt * 1.5))

        if gaze == GazeZone.EYES_CLOSED or attention in (AttentionState.MICROSLEEP, AttentionState.INCAPACITATED):
            self.eye_closure_timer += dt
        else:
            self.eye_closure_timer = max(0.0, self.eye_closure_timer - (dt * 2.0))

        if not hands_on_wheel:
            self.hands_off_timer += dt
        else:
            self.hands_off_timer = max(0.0, self.hands_off_timer - (dt * 2.0))

        # Check driver recovery (attentive + hands on wheel)
        driver_recovering = (
            self.is_gaze_attentive(gaze) and 
            hands_on_wheel and 
            attention == AttentionState.ATTENTIVE and 
            gaze != GazeZone.EYES_CLOSED
        )

        if driver_recovering:
            self.recovery_timer += dt
        else:
            self.recovery_timer = 0.0

        # Decrement transient brake jerk timer
        if self.brake_jerk_remaining > 0.0:
            self.brake_jerk_remaining = max(0.0, self.brake_jerk_remaining - dt)

        # 2. Driver recovery handling
        if not self.mrm_latched and self.recovery_timer >= 0.5:
            # Driver regained situational awareness
            self.current_level = InterventionLevel.LEVEL_0_PASSIVE
            self.distraction_timer = 0.0
            self.eye_closure_timer = 0.0
            self.hands_off_timer = 0.0
            self.brake_jerk_remaining = 0.0

        # 3. Escalation Logic
        if not self.mrm_latched:
            # Emergency Fast-Path: Driver incapacitation / collapse
            if attention == AttentionState.INCAPACITATED or driver_present == DriverPresence.ABSENT:
                if self.eye_closure_timer >= 2.0 or not is_moving:
                    self.current_level = InterventionLevel.LEVEL_5_MRM_STOP
                    self.mrm_latched = True
                else:
                    self.current_level = InterventionLevel.LEVEL_4_BRAKE_JERK
                    if self.brake_jerk_remaining <= 0.0:
                        self.brake_jerk_remaining = 0.35

            # Standard 5-Tier Escalation Progression
            elif is_moving:
                if self.eye_closure_timer >= 4.5 or self.distraction_timer >= 8.0:
                    self.current_level = InterventionLevel.LEVEL_5_MRM_STOP
                    self.mrm_latched = True
                elif self.eye_closure_timer >= 3.0 or self.distraction_timer >= 6.0:
                    if self.current_level < InterventionLevel.LEVEL_4_BRAKE_JERK:
                        self.brake_jerk_remaining = 0.30  # Trigger 300ms brake jerk
                    self.current_level = InterventionLevel.LEVEL_4_BRAKE_JERK
                elif self.eye_closure_timer >= 2.0 or self.distraction_timer >= 4.5 or drowsiness >= DrowsinessLevel.SEVERE_DROWSY:
                    self.current_level = InterventionLevel.LEVEL_3_HAPTIC
                elif self.eye_closure_timer >= 1.2 or self.distraction_timer >= 3.0 or drowsiness >= DrowsinessLevel.MODERATE_DROWSY:
                    self.current_level = InterventionLevel.LEVEL_2_ACOUSTIC
                elif self.distraction_timer >= 1.8 or drowsiness >= DrowsinessLevel.MILD_DROWSY or self.hands_off_timer >= 5.0:
                    self.current_level = InterventionLevel.LEVEL_1_VISUAL
                else:
                    self.current_level = InterventionLevel.LEVEL_0_PASSIVE

        # 4. Synthesize Actuator Signals from Level
        level = self.current_level
        visual = 0
        acoustic = 0
        haptic_wheel = False
        seatbelt_tug = False
        brake_jerk = False
        hazard = False
        mrm = False
        ecall = False
        target_decel = 0.0

        if level == InterventionLevel.LEVEL_1_VISUAL:
            visual = 1  # Yellow caution icon
            desc = "Visual Alert: Amber warning displayed on cluster/HUD"
        elif level == InterventionLevel.LEVEL_2_ACOUSTIC:
            visual = 2  # Flashing amber/red
            acoustic = 1 # Warning chime (75dB)
            desc = "Acoustic Warning: Chime active + Cluster alert"
        elif level == InterventionLevel.LEVEL_3_HAPTIC:
            visual = 2
            acoustic = 2 # Urgent alarm (80dB)
            haptic_wheel = True # 100Hz steering wheel shake
            seatbelt_tug = True # Motorized pretensioner pulse
            desc = "Haptic Intervention: Steering wheel vibration + Seatbelt tug"
        elif level == InterventionLevel.LEVEL_4_BRAKE_JERK:
            visual = 2
            acoustic = 2
            haptic_wheel = True
            seatbelt_tug = True
            hazard = True
            brake_jerk = (self.brake_jerk_remaining > 0.0)
            target_decel = 2.5 if brake_jerk else 0.0
            desc = "Tactile Intervention: Brake jerk pulse (-2.5 m/s^2) + Hazard lights"
        elif level == InterventionLevel.LEVEL_5_MRM_STOP:
            visual = 2
            acoustic = 2
            haptic_wheel = True
            hazard = True
            mrm = True
            ecall = True
            target_decel = 2.0 # Controlled deceleration to safe stop
            desc = "CRITICAL: Level 5 MRM Active. Autonomous Safe Stop + eCall SOS triggered"
        else:
            desc = "Passive Monitoring: Driver attentive"

        return EscalationOutput(
            intervention_level=level,
            visual_alert=visual,
            acoustic_alert=acoustic,
            haptic_wheel=haptic_wheel,
            seatbelt_tug=seatbelt_tug,
            brake_jerk=brake_jerk,
            hazard_flash=hazard,
            mrm_takeover=mrm,
            ecall_trigger=ecall,
            target_decel_mps2=target_decel,
            state_description=desc
        )
