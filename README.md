# Robotaxi L4 Occupant Monitoring System (OMS) — Unresponsive Passenger Deep Dive

A conceptual, software-simulated Occupant Monitoring System (OMS) and multi-tier intervention architecture designed for SAE Level 4 driverless robotaxis, specifically addressing the medical emergency and unresponsive passenger scenario.

---

> [!NOTE]
> **Conceptual System Architecture**: All CAN bus frames, network topologies, and escalation signals described in this repository are **purely conceptual software simulations**. This project does not deploy physical CAN hardware transceivers, electrical bus lines, voltages, or hardware-enforced electrical timing. All communication is simulated via in-memory data structures and virtual loopback interfaces.

---

## 1. System Overview & The L4 Problem

In SAE Level 4 autonomous robotaxis, there is no driver or steering wheel inside the vehicle. The traditional Driver Monitoring System (DMS) is no longer sufficient; the vehicle itself must bear legal and ethical responsibility for passenger health, security, and cabin safety through an **Occupant Monitoring System (OMS)**.

### The Core Scenario: Unresponsive Passenger
When a robotaxi arrives at its drop-off destination with doors unlocked, an occupant who remains completely motionless on the rear seat presents a critical dilemma: Are they asleep, or are they experiencing a life-threatening medical event (e.g., cardiac arrest, stroke)? Without a human driver to intervene, the vehicle's perception and decision stack must autonomously evaluate the situation and escalate appropriate countermeasures.

```
+----------------------------------------------------------------------------------------------------+
|                                    CONCEPTUAL VEHICLE ARCHITECTURE                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  +---------------------------+        +--------------------------+                                 |
|  |   Cabin Wide-Angle Camera |        |  Passenger Face Camera   |                                 |
|  +-------------┬-------------+        +------------┬-------------+                                 |
|                │                                   │                                               |
|                ▼                                   ▼                                               |
|  +---------------------------------------------------------------+                                 |
|  |               OMS Vision Perception Stack                     |                                 |
|  |   - Body 26-pt Keypoint Estimation (Halpe-26 Topology)        |                                 |
|  |   - Facial Mesh Landmark Tracking (50-point Contour)          |                                 |
|  +-----------------------------┬---------------------------------+                                 |
|                                │ Fused Metric Extraction                                           |
|                                ▼                                                                   |
|  +---------------------------------------------------------------+                                 |
|  |             Multi-Modal Geometric Decision Engine             |                                 |
|  |   - Eye Aspect Ratio (EAR) < 0.20 (Closed / Unresponsive)     |                                 |
|  |   - Trunk Lean Angle θ > 40° (Slumped Torso Posture)          |                                 |
|  |   - Temporal Stillness Tracker (Stillness Duration >= t)      |                                 |
|  +-----------------------------┬---------------------------------+                                 |
|                                │ Conceptual Protocol Dispatch                                      |
|                                ▼                                                                   |
|                   ===================================== Simulated Vehicle Bus                      |
|                   Virtual CAN Protocol: DIP_CMD (0x0F0) / OMS_STS (0x130)                          |
|                   =====================================                                            |
|                        │            │           │            │                                     |
|                        ▼            ▼           ▼            ▼                                     |
|                   +---------+  +---------+  +---------+  +---------+                               |
|                   | Cabin   |  | Exterior|  | Fleet   |  | Remote  |                               |
|                   | Audio   |  | Display |  | Tele-op |  | Unlock  |                               |
|                   | Lights  |  | Siren   |  | Van     |  | & eCall |                               |
|                   +---------+  +---------+  +---------+  +---------+                               |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Perception & Mathematical Foundations

Instead of uninterpretable black-box models, the perception module derives interpretable geometric indicators in real time:

1. **Eye Aspect Ratio (EAR)**:
   $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
   *Academic Citation*: Soukupova & Cech, *"Real-Time Eye Blink Detection using Facial Landmarks"*, CVWW 2016. Identifies sustained eye closure and loss of consciousness ($\text{EAR} < 0.20$).

2. **Trunk Lean Angle ($\theta$)**:
   $$\theta = \arccos\left(\frac{\Delta y}{\sqrt{\Delta x^2 + \Delta y^2}}\right)$$
   Evaluates the vector from `HipCenter` to `Neck` relative to the vertical axis using Euclidean vector projection. An angle $\theta > 40^\circ$ signifies a slumped or collapsed upper body.

3. **Temporal Stillness Integration**:
   Tracks elapsed duration since trip completion with open doors where keypoint displacement remains zero ($\Delta \text{pose} \approx 0$).

---

## 3. 4-Tier Escalation Hierarchy

The conceptual intervention controller escalates systematically across 4 distinct domains:

| Escalation Tier | Target Domain | Trigger Criteria | Actuator Response |
| :--- | :--- | :--- | :--- |
| **Level 1** | Cabin Interior | Stillness $\ge 60\text{s}$ | 100% full dome lighting, high-volume audible wakeup chime, seat HMI display notification. |
| **Level 2** | Passerby / Exterior | Stillness $\ge 120\text{s}$ | Exterior glass HUD displays *"CẦN TRỢ GIÚP Y TẾ"*, external pedestrian speaker broadcast SOS. |
| **Level 3** | Fleet Tele-Operation | Stillness $\ge 150\text{s}$ | Two-way audio patch to remote operator console, dispatch of mobile Field Support Van. |
| **Level 4** | Emergency Services | Stillness $\ge 180\text{s}$ | Autonomous eCall 115 emergency dispatch with telemetry, automatic remote door unlock for first responders. |

---

## 4. Simulated Vehicle Message Protocol

The communication layer models standard automotive framing principles via software abstractions:

- **`DIP_InterventionCommand` (`0x0F0`)**: 8-byte command payload specifying intervention level, alert actuation triggers, target deceleration, alive counter, and SAE J1850 CRC-8 checksum.
- **`OMS_Status` (`0x130`)**: 8-byte cabin telemetry payload providing occupancy counts, seatbelt status, and anomaly flags.

---

## 5. Verification & Execution

### 1. Interactive Web Dashboard
Open `demo.html` in any modern web browser to view the real-time 2D cabin canvas, 50-point face landmark mesh, metric gauges, and simulated CAN bus telemetry stream.

### 2. Slide Presentation Deck
Open `slides.html` in a web browser for the Reveal.js academic presentation covering SAE automation levels (L2 to L4), algorithmic derivations, and the 4-tier escalation protocol.

### 3. Running the Simulation
```bash
python3 scripts/simulate_oms.py
```

### 4. Running the Unit Test Suite
```bash
python3 -m unittest discover -s tests -v
```
