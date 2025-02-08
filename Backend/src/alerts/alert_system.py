import pygame
import logging
import numpy as np
from datetime import datetime


class SimpleAlertSystem:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize pygame for sound
        try:
            pygame.mixer.init()
            self.sound_enabled = True
            self.logger.info("Sound system initialized successfully")
        except Exception as e:
            self.sound_enabled = False
            self.logger.warning(f"Sound system initialization failed: {str(e)}")

        # Create a log file for detections
        self.log_file = f"deepfake_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    def trigger_alert(self, confidence_score, frame_number=None):
        """
        Trigger alert when a deepfake is detected

        Args:
            confidence_score (float): Detection confidence score (0-1)
            frame_number (int, optional): Frame number where deepfake was detected
        """
        # Log the detection
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = (
            f"DEEPFAKE DETECTED at {timestamp}\n"
            f"Confidence Score: {confidence_score:.2%}\n"
            f"Frame Number: {frame_number if frame_number else 'N/A'}\n"
            f"------------------------\n"
        )

        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(alert_message)
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {str(e)}")

        # Play alert sound
        if self.sound_enabled:
            try:
                self._play_beep()
            except Exception as e:
                self.logger.error(f"Failed to play alert sound: {str(e)}")

        # Print to console for immediate feedback
        print("\n⚠️ " + alert_message)

    def _play_beep(self):
        """Play a simple beep sound"""
        frequency = 440  # Hz (A4 note)
        duration = 500  # milliseconds

        # Create a simple beep sound using pygame
        pygame.mixer.quit()  # Reset the mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        pygame.mixer.Sound(np.random.rand(duration)).play()

        # Add a small delay to ensure the sound plays
        pygame.time.delay(duration)


# Example usage
if __name__ == "__main__":
    # Initialize the alert system
    alert_system = SimpleAlertSystem()

    # Test the alert system
    alert_system.trigger_alert(
        confidence_score=0.95,
        frame_number=1234
    )