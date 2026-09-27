# Driver Intervention Program: DMS/OMS Escalation via CAN_H & CAN_L

Deterministic, ISO 26262 / Euro NCAP compliant automotive subsystem bridging cabin sensing (DMS / OMS) to chassis actuators (HMI, BCM, EPS, ESP, TCU) over physical high-speed CAN (`canhi` and `canlo`).

---

## 1. System Architecture & Safety Partitioning

```
+----------------------------------------------------------------------------------------------------+
|                                    VEHICLE CABIN & COMPUTE ARCHITECTURE                            |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  +---------------------------+        +--------------------------+                                 |
|  |  DMS Camera (IR/NIR 60Hz) |        | OMS Cabin Camera (RGB-IR)|                                 |
|  +-------------┬-------------+        +------------┬-------------+                                 |
|                │                                   │                                               |
|                ▼                                   ▼                                               |
|  +---------------------------------------------------------------+                                 |
|  |     Vision Perception ECU (ASIL-B / Linux / QNX / Orin / 8155)|                                 |
|  |     - Eye Closure / PERCLOS / Head Pose / Gaze Zone           |                                 |
|  |     - Occupancy / Belt Status / Driver Incapacitation         |                                 |
|  +-----------------------------┬---------------------------------+                                 |
|                                │ UART / SPI / CAN Controller                                       |
|                                ▼                                                                   |
|  +---------------------------------------------------------------+                                 |
|  |     CAN Physical Transceiver (e.g. NXP TJA1042 / TI TCAN1042)  |                                 |
|  +-----------------------------┬---------------------------------+                                 |
|                                │                                                                   |
|                   =============┴======================= Vehicle Safety CAN Bus                      |
|                   CAN_H (canhi): 2.5V (rec) -> 3.5V (dom)                                          |
|                   CAN_L (canlo): 2.5V (rec) -> 1.5V (dom)  [120 Ohm Split Termination]             |
|                   =====================================                                            |
|                        │            │           │            │                                     |
|                        ▼            ▼           ▼            ▼                                     |
|                   +---------+  +---------+  +---------+  +---------+                               |
|                   |   HMI   |  |   BCM   |  | ESP/ESC |  |   TCU   |                               |
|                   | Cluster |  | Seatbelt|  |  Brake  |  |  eCall  |                               |
|                   |  Alerts |  | Hazards |  |   MRM   |  |   SOS   |                               |
|                   +---------+  +---------+  +---------+  +---------+                               |
+----------------------------------------------------------------------------------------------------+
```

### Automotive Safety Reality Check (ISO 26262 & SOTIF)
1. **ASIL Separation**: Vision neural nets running on Linux/QNX compute SoCs are at most **ASIL-B**. They must **never** directly command hydraulic braking or steering line actuators.
2. **Safety Gateway / Supervisor Interlock**: The escalation state machine formats an E2E-protected command message (`DIP_InterventionCommand`, ID `0x0F0`) with a 4-bit rolling counter and SAE J1850 CRC-8. Downstream chassis ECUs (ASIL-D ESP/EPS) validate the E2E profile before executing physical deceleration or steering takeover.

---

## 2. Physical Layer Specification (`canhi` and `canlo`)

Compliant with **ISO 11898-2 (High-Speed CAN)** @ 500 kbps:

| Bus State | Logic Bit | `CAN_H` (`canhi`) | `CAN_L` (`canlo`) | Differential $V_{\text{diff}}$ ($V_H - V_L$) | Line Drive State |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Recessive** | `1` | **2.5 V** | **2.5 V** | **0.0 V** | High-impedance passive termination (120 $\Omega$) |
| **Dominant** | `0` | **3.5 V** | **1.5 V** | **+2.0 V** | Transceiver active push-pull drive |

*Termination*: Two 120 $\Omega$ resistors placed at extreme ends of the twisted pair network to prevent RF reflections and signal ringing.

---

## 3. Escalation State Progression (Euro NCAP / UN ECE R157 MRM)

| Escalation Level | Trigger Criteria | Actuator Response | CAN Signals Triggered |
| :--- | :--- | :--- | :--- |
| **Level 0 (Passive)** | Attentive driving | Nominal logging | `Level=0`, All actuators off |
| **Level 1 (Visual)** | Gaze off-road $\ge 2.0\text{s}$ or Drowsiness $\ge$ Mild | Amber warning on cluster/HUD | `VisualAlert=1` |
| **Level 2 (Acoustic)** | Gaze off-road $\ge 3.5\text{s}$ or Eyes closed $\ge 1.5\text{s}$ | 75-80 dBA pulsed acoustic chime | `VisualAlert=2`, `AcousticAlert=1` |
| **Level 3 (Haptic)** | Gaze off-road $\ge 5.0\text{s}$ or Eyes closed $\ge 2.5\text{s}$ | 100 Hz steering wheel vibration + Motorized seatbelt tensioner tug | `HapticWheel=1`, `SeatbeltTug=1`, `AcousticAlert=2` |
| **Level 4 (Tactile)** | Gaze off-road $\ge 6.5\text{s}$ or Eyes closed $\ge 3.5\text{s}$ | 300ms warning brake pulse ($a = -2.5\text{ m/s}^2$) + Hazard flashers | `BrakeJerk=1`, `HazardFlash=1`, `Decel=2.5` |
| **Level 5 (MRM Safe Stop)** | Gaze off-road $\ge 8.0\text{s}$ or Eyes closed $\ge 5.0\text{s}$ or Incapacitated | Minimum Risk Maneuver: controlled stop ($a = -2.0\text{ m/s}^2$), hazard lights, park lock, eCall SOS | `MRM_Takeover=1`, `eCallTrigger=1`, `Decel=2.0` |

---

## 4. CAN Frame Matrix & DBC Specification

Refer to [`protocol/dms_intervention.dbc`](protocol/dms_intervention.dbc).

### High-Priority Command: `DIP_InterventionCommand` (`0x0F0` / 240 dec)
- **Period**: 10 ms
- **Payload Layout (8 bytes)**:
  - `Byte 0`: `[0..2]` Intervention Level | `[3..4]` Visual Alert | `[5..6]` Acoustic Alert | `[7]` Haptic Wheel
  - `Byte 1`: `[0]` Seatbelt Tug | `[1]` Brake Jerk | `[2]` Hazard Flash | `[3]` MRM Active | `[4]` eCall Trigger
  - `Byte 2`: Target Deceleration ($a_{\text{decel}} = \text{RAW} \times 0.05\text{ m/s}^2$)
  - `Byte 3..5`: Reserved (`0x00`)
  - `Byte 6`: `[0..3]` Rolling Alive Counter ($0 \to 15$)
  - `Byte 7`: SAE J1850 CRC-8 Checksum

---

## 5. Verification & Execution

### Running the Test Suite
```bash
python3 -m unittest discover -s tests -v
```

### Running the Real-Time Escalation Simulation
```bash
python3 scripts/simulate_escalation.py
```
