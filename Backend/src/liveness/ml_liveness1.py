import tensorflow as tf
import dlib
import numpy as np
import cv2
import os


class LivenessDetector:
    def __init__(self, landmark_predictor_path='shape_predictor_68_face_landmarks.dat'):
        # Load pre-trained facial landmark predictor
        self.landmark_predictor = dlib.shape_predictor(landmark_predictor_path)

        # Build ML model for liveness detection
        self.model = self.create_liveness_model()

    def create_liveness_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(100, 100, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: blink, nod, tilt
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def detect_landmarks(self, frame):
        # Convert frame to dlib image
        dlib_frame = dlib.cv_image_to_cv_mat(frame)

        # Detect face
        detector = dlib.get_frontal_face_detector()
        faces = detector(dlib_frame)

        if len(faces) == 0:
            return None

        # Get landmarks for first face
        shape = self.landmark_predictor(dlib_frame, faces[0])
        landmarks = np.array([(p.x, p.y) for p in shape.parts()])

        return landmarks

    def extract_features(self, frame, landmarks):
        # Placeholder for feature extraction
        # You'll need to implement actual feature extraction
        features = np.random.rand(1, 100, 100, 3)  # Example random features
        return features

    def classify_liveness(self, features):
        # Classify liveness
        prediction = self.model.predict(features)
        return np.argmax(prediction)

    def process_video(self, video_path):
        # Open video file
        cap = cv2.VideoCapture(video_path)
        liveness_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect landmarks
            landmarks = self.detect_landmarks(frame)

            if landmarks is not None:
                # Extract features
                features = self.extract_features(frame, landmarks)

                # Classify liveness
                result = self.classify_liveness(features)
                liveness_results.append(result)

        cap.release()
        return liveness_results


def load_dataset(directory):
    real_videos = []
    fake_videos = []

    real_folder = os.path.join(directory, 'real')
    fake_folder = os.path.join(directory, 'fake')

    real_videos = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith('.mp4')]
    fake_videos = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith('.mp4')]

    return real_videos, fake_videos


# Example usage
def main():
    # Path to your dataset
    directory_path = directory_path = r"C:\Users\SHALINI\Desktop\python programs\New folder\datasets"



    # Load videos
    real_videos, fake_videos = load_dataset(directory_path)

    # Initialize detector
    liveness_detector = LivenessDetector()

    # Process some videos
    for video in real_videos[:5]:  # First 5 real videos
        results = liveness_detector.process_video(video)
        print(f"Liveness results for {video}: {results}")


if __name__ == "__main__":
    main()