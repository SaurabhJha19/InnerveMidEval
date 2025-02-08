import cv2
import dlib
from mtcnn import MTCNN
from src.detection import face_detection
from src.liveness import liveness_detector
from src.alerts import alert_system
from src.logging import logger
from flask import Flask, render_template


class DeepfakeDetector:
    def _init_(self):
        # Initialize core components
        self.face_detector = face_detection()
        self.liveness_detector = liveness_detector()
        self.alert_system = alert_system()
        self.logger = logger()

    def run_detection(self, frame):
        # Main detection pipeline
        faces = self.face_detector.detect(frame)
        for face in faces:
            is_real = self.liveness_detector.check_liveness(face)
            if not is_real:
                self.alert_system.trigger_alert()
                self.logger.log_incident()