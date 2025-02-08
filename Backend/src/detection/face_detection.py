import cv2
import dlib
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        # Load detectors
        self.opencv_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.mtcnn_detector = MTCNN()

    def detect_faces(self, frame):
        """ Detect faces using MTCNN, then Dlib, then OpenCV (fallback). """
        faces = []

        # Convert frame to RGB (MTCNN & Dlib need RGB, OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1️⃣ Try MTCNN first (most accurate)
        try:
            mtcnn_faces = self.mtcnn_detector.detect_faces(frame_rgb)
            if mtcnn_faces:
                for face in mtcnn_faces:
                    x, y, width, height = face["box"]
                    faces.append((x, y, x + width, y + height))
                return faces
        except:
            pass  # If MTCNN fails, move to the next method

        # 2️⃣ Try Dlib as a fallback
        dlib_faces = self.dlib_detector(frame_rgb)
        if dlib_faces:
            for face in dlib_faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                faces.append((x, y, x + w, y + h))
            return faces

        # 3️⃣ Try OpenCV Haar Cascade as the last option
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        opencv_faces = self.opencv_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in opencv_faces:
            faces.append((x, y, x + w, y + h))

        return faces
