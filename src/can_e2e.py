"""
E2E (End-to-End) Protection for Automotive CAN Frames.
Compliant with AUTOSAR E2E Profile 01 / SAE J1850 CRC-8 standard.
"""

def calculate_crc8_sae_j1850(data: bytes) -> int:
    """
    Computes SAE J1850 CRC-8 checksum.
    Polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x1D)
    Initial value: 0xFF
    Final XOR: 0xFF
    """
    crc = 0xFF
    poly = 0x1D
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc ^ 0xFF


class E2ETracker:
    """
    Tracks rolling alive counter (0-15) and validates incoming E2E frames.
    """
    def __init__(self, max_counter: int = 15):
        self.counter = 0
        self.max_counter = max_counter
        self.last_received_counter = -1

    def next_counter(self) -> int:
        c = self.counter
        self.counter = (self.counter + 1) % (self.max_counter + 1)
        return c

    def validate_frame(self, payload: bytes) -> bool:
        """
        Validates payload length (8 bytes), CRC-8 (byte 7), and alive counter progression (byte 6).
        """
        if len(payload) != 8:
            return False
        
        expected_crc = calculate_crc8_sae_j1850(payload[:7])
        actual_crc = payload[7]
        if expected_crc != actual_crc:
            return False

        received_counter = payload[6] & 0x0F
        if self.last_received_counter != -1:
            expected_next = (self.last_received_counter + 1) % (self.max_counter + 1)
            # Allow at most 1 missed frame in lossy conditions
            if received_counter != expected_next and received_counter != (expected_next + 1) % (self.max_counter + 1):
                # Out of sequence
                self.last_received_counter = received_counter
                return False
        
        self.last_received_counter = received_counter
        return True
