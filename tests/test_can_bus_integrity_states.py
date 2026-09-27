"""
Comprehensive Unit Tests for CAN Bus Integrity Verification Across:
1. HEALTHY (Error Active, nominal CRC-8 & alive counter progression)
2. PARTIAL SHUTDOWN (Error Passive, bit corruption rejection, packet loss detection)
3. COMPLETE SHUTDOWN (Bus-Off, total silence timeout, socket loss, ASIL-D safe-state latching)
"""

import unittest
import time
from src.can_bus_interface import (
    CANBusInterface, CANMessage, CANBusHealthState, CANBusSupervisor
)
from src.can_frames import (
    pack_dip_intervention_command, unpack_dip_intervention_command,
    pack_oms_status, unpack_oms_status
)
from src.can_e2e import calculate_crc8_sae_j1850, E2ETracker
from src.states import InterventionLevel, DriverPresence, BeltStatus


class TestCANBusIntegrityStates(unittest.TestCase):

    def setUp(self):
        self.bus = CANBusInterface(virtual_fallback=True)
        self.supervisor = CANBusSupervisor(timeout_threshold_s=0.25)
        self.e2e_tracker = E2ETracker(max_counter=15)

    def tearDown(self):
        self.bus.close()

    # =========================================================================
    # 1. HEALTHY STATE VERIFICATION
    # =========================================================================
    def test_healthy_state_nominal_transmission(self):
        """Verifies clean transmission, valid SAE J1850 CRC-8, and sequential alive counters."""
        for i in range(16):
            alive_cnt = self.e2e_tracker.next_counter()
            payload = pack_dip_intervention_command(
                intervention_level=InterventionLevel.LEVEL_1_VISUAL,
                visual_alert=1,
                acoustic_alert=0,
                haptic_wheel=False,
                seatbelt_tug=False,
                brake_jerk=False,
                hazard_flash=False,
                mrm_takeover=False,
                ecall_trigger=False,
                target_decel_mps2=0.0,
                alive_counter=alive_cnt
            )
            # Send
            sent = self.bus.send(0x0F0, payload)
            self.assertTrue(sent)
            self.supervisor.record_tx()

            # Receive
            msg = self.bus.recv()
            self.assertIsNotNone(msg)
            self.assertEqual(len(msg.data), 8)

            # E2E validation
            is_valid_e2e = self.e2e_tracker.validate_frame(msg.data)
            self.assertTrue(is_valid_e2e, f"Frame {i} should be valid E2E")

            # Check unpack
            unpacked = unpack_dip_intervention_command(msg.data)
            self.assertEqual(unpacked["alive_counter"], alive_cnt)
            self.supervisor.record_rx_success(timestamp=time.time())

        # Health state assertion
        health = self.supervisor.evaluate_health(bus_interface=self.bus)
        self.assertEqual(health, CANBusHealthState.HEALTHY)
        self.assertEqual(self.supervisor.crc_errors, 0)
        self.assertEqual(self.supervisor.dropped_frames, 0)
        self.assertEqual(self.supervisor.valid_frames, 16)

    # =========================================================================
    # 2. PARTIAL SHUTDOWN / DEGRADATION VERIFICATION
    # =========================================================================
    def test_partial_shutdown_bit_corruption_rejection(self):
        """Verifies that corrupted frames fail CRC8 check and transition supervisor to PARTIAL_SHUTDOWN."""
        # 1. Send valid frame
        valid_payload = pack_dip_intervention_command(
            intervention_level=InterventionLevel.LEVEL_0_PASSIVE,
            visual_alert=0, acoustic_alert=0, haptic_wheel=False,
            seatbelt_tug=False, brake_jerk=False, hazard_flash=False,
            mrm_takeover=False, ecall_trigger=False, target_decel_mps2=0.0,
            alive_counter=1
        )
        self.bus.send(0x0F0, valid_payload)
        msg1 = self.bus.recv()
        self.supervisor.record_rx_success(time.time())

        # 2. Inject corrupted bit in payload
        corrupted_payload = bytearray(valid_payload)
        corrupted_payload[2] ^= 0x55  # Tamper with deceleration byte
        with self.assertRaises(ValueError):
            unpack_dip_intervention_command(bytes(corrupted_payload))

        self.supervisor.record_crc_error()

        # 3. Evaluate health: should be PARTIAL_SHUTDOWN
        health = self.supervisor.evaluate_health(bus_interface=self.bus)
        self.assertEqual(health, CANBusHealthState.PARTIAL_SHUTDOWN)
        self.assertEqual(self.supervisor.crc_errors, 1)

    def test_partial_shutdown_frame_loss_detection(self):
        """Verifies that dropped frames cause alive counter jumps detected by E2E tracker."""
        tracker = E2ETracker(max_counter=15)
        # Receive counter 0
        p0 = pack_dip_intervention_command(
            InterventionLevel.LEVEL_0_PASSIVE, 0, 0, False, False, False, False, False, False, 0.0, 0
        )
        self.assertTrue(tracker.validate_frame(p0))

        # Simulate 3 dropped frames (counter jumps from 0 to 4)
        p4 = pack_dip_intervention_command(
            InterventionLevel.LEVEL_0_PASSIVE, 0, 0, False, False, False, False, False, False, 0.0, 4
        )
        is_valid = tracker.validate_frame(p4)
        self.assertFalse(is_valid, "E2E tracker must flag skipped counter jump (frame drop)")

        self.supervisor.record_frame_drop(count=3)
        health = self.supervisor.evaluate_health(bus_interface=self.bus)
        self.assertEqual(health, CANBusHealthState.PARTIAL_SHUTDOWN)
        self.assertEqual(self.supervisor.dropped_frames, 3)

    # =========================================================================
    # 3. COMPLETE SHUTDOWN / BUS-OFF VERIFICATION
    # =========================================================================
    def test_complete_shutdown_bus_off_simulation(self):
        """Verifies simulated Bus-Off state rejects all I/O and triggers ASIL-D safe-state defaults."""
        self.bus.trigger_bus_off()
        self.assertTrue(self.bus.is_bus_off)

        # Transmissions must fail in Bus-Off
        payload = bytes([0x00] * 8)
        sent = self.bus.send(0x0F0, payload)
        self.assertFalse(sent)

        # Receive must return None
        received = self.bus.recv()
        self.assertIsNone(received)

        # Supervisor must detect COMPLETE_SHUTDOWN
        health = self.supervisor.evaluate_health(bus_interface=self.bus)
        self.assertEqual(health, CANBusHealthState.COMPLETE_SHUTDOWN)

        # Verify ASIL-D Safe-State defaults
        failsafe = self.supervisor.get_failsafe_actuation()
        self.assertTrue(failsafe["failsafe_latched"])
        self.assertTrue(failsafe["mrm_takeover"])
        self.assertGreaterEqual(failsafe["target_decel_mps2"], 1.5)
        self.assertTrue(failsafe["hazard_flash"])
        self.assertTrue(failsafe["emergency_door_unlock"])
        self.assertEqual(failsafe["interior_dome_light_pct"], 100)

    def test_complete_shutdown_message_silence_timeout(self):
        """Verifies total silence > 250ms triggers COMPLETE_SHUTDOWN."""
        now = time.time()
        self.supervisor.record_rx_success(timestamp=now)

        # Health right after frame: Healthy
        self.assertEqual(self.supervisor.evaluate_health(current_time=now + 0.05), CANBusHealthState.HEALTHY)

        # Health after 300ms silence (timeout is 250ms): COMPLETE_SHUTDOWN
        health_timeout = self.supervisor.evaluate_health(current_time=now + 0.30)
        self.assertEqual(health_timeout, CANBusHealthState.COMPLETE_SHUTDOWN)

    def test_complete_shutdown_socket_close(self):
        """Verifies closing interface safely closes bus and latches safe state."""
        self.bus.close()
        self.assertTrue(self.bus.is_bus_off)
        self.assertIsNone(self.bus.sock)

        health = self.supervisor.evaluate_health(bus_interface=self.bus)
        self.assertEqual(health, CANBusHealthState.COMPLETE_SHUTDOWN)


if __name__ == '__main__':
    unittest.main()
