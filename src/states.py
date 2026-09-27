from enum import IntEnum

class InterventionLevel(IntEnum):
    LEVEL_0_PASSIVE = 0         # Normal monitoring
    LEVEL_1_VISUAL = 1          # Visual alert on Instrument Cluster / HUD
    LEVEL_2_ACOUSTIC = 2        # Chime / Audio warning + Visual
    LEVEL_3_HAPTIC = 3          # Steering wheel vibration + Seatbelt motorized tug
    LEVEL_4_BRAKE_JERK = 4      # Short brake pulse (-0.2g, 300ms) + Hazard lights
    LEVEL_5_MRM_STOP = 5        # Minimum Risk Maneuver (Autonomous safe stop + eCall)

class AttentionState(IntEnum):
    ATTENTIVE = 0
    DISTRACTED_LOW = 1
    DISTRACTED_HIGH = 2
    DROWSY = 3
    MICROSLEEP = 4
    INCAPACITATED = 5

class GazeZone(IntEnum):
    ROAD_AHEAD = 0
    LEFT_MIRROR = 1
    RIGHT_MIRROR = 2
    CLUSTER = 3
    INFOTAINMENT = 4
    PHONE_DOWN = 5
    EYES_CLOSED = 6
    UNKNOWN = 7

class DrowsinessLevel(IntEnum):
    ALERT = 0
    QUESTIONABLE = 1
    MILD_DROWSY = 2
    MODERATE_DROWSY = 3
    SEVERE_DROWSY = 4
    SLEEP_ONSET = 5

class DriverPresence(IntEnum):
    ABSENT = 0
    PRESENT = 1

class BeltStatus(IntEnum):
    UNBUCKLED = 0
    BUCKLED = 1
